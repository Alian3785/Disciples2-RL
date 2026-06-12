# visualize_campaign.py
"""
Визуализация прохождения кампании обученным агентом.

Показывает:
1. Карту 20×20 с перемещением агента
2. Бои с врагами (полная визуализация из visualize.py)
3. Общий прогресс кампании
"""

import os
import time
import re
import math
import argparse
import textwrap
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch

try:
    from IPython.display import display
except ImportError:
    display = None

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from campaign_env import CampaignEnv
from grid import DEFAULT_GRID_SIZE, DEFAULT_HERO_GRID_POSITION
from train_campaign import REWARD_CONFIG
from enemy_configs import ENEMY_DESCRIPTIONS
from battle_env import (
    TARGET_POSITIONS,
    DEFEND_ACTION_INDEX,
    WAIT_ACTION_INDEX,
    RUN_AWAY_ACTION_INDEX,
    FIRST_HERO_ITEM_ACTION_START,
    SECOND_HERO_ITEM_ACTION_START,
    FIRST_HERO_ITEM_ENEMY_ACTION_START,
    SECOND_HERO_ITEM_ENEMY_ACTION_START,
    HERO_ITEM_TARGET_SLOTS,
    DEFEND_ARMOR_BONUS,
)
from console_encoding import setup_utf8_console

try:
    from tools.inspect_sg_map import (
        RENDER_OBJECT_SPECS,
        extract_render_objects,
        parse_settlement_objects,
        parse_stacks,
    )
except Exception:
    RENDER_OBJECT_SPECS = {}
    extract_render_objects = None
    parse_settlement_objects = None
    parse_stacks = None


setup_utf8_console()


DEFAULT_SCENARIO_PATH = Path(
    "D:/Disciples II "
    "\u0412\u043e\u0441\u0441\u0442\u0430\u043d\u0438\u0435 "
    "\u042d\u043b\u044c\u0444\u043e\u0432/Exports/A Return To Simpler Times.sg"
)
DEFAULT_MAP_PNG_PATH = Path("./outputs/a_return_to_simpler_times_entries.png")


def _existing_path_or_none(path_value: str | os.PathLike | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    return str(path) if path.exists() else None


def mask_fn(env: CampaignEnv) -> np.ndarray:
    return env.compute_action_mask()


GRID_MOVE_ACTION_NAMES = ["↑", "↓", "←", "→", "↖", "↗", "↙", "↘", "⊙"]


def _format_grid_action_label(action: int, info: dict | None = None) -> str:
    info = dict(info or {})
    if info.get("spell_cast_action"):
        spell_description = str(info.get("spell_description") or info.get("spell_key") or "spell")
        target_enemy_id = info.get("target_enemy_id")
        target_suffix = ""
        if target_enemy_id is not None:
            target_suffix = f" -> E{int(target_enemy_id)}"
        return f"Spell: {spell_description}{target_suffix}"
    if info.get("spell_shop_buy_action"):
        spell_name = str(info.get("spell_shop_buy_spell") or "spell").strip()
        site_name = str(info.get("spell_shop_buy_site") or "").strip()
        if site_name:
            return f"Spell shop: {spell_name} ({site_name})"
        return f"Spell shop: {spell_name}"
    if info.get("mercenary_hire_action"):
        unit_name = str(
            info.get("mercenary_hire_unit_name")
            or info.get("hired_unit_name")
            or info.get("hire_unit_name")
            or "unit"
        ).strip()
        site_name = str(info.get("mercenary_hire_site") or "").strip()
        if site_name:
            return f"Merc: {unit_name} ({site_name})"
        return f"Merc: {unit_name}"
    if info.get("trainer_action"):
        unit_name = str(info.get("trainer_unit_name") or "unit").strip()
        xp_gained = int(info.get("trainer_xp_gained", 0) or 0)
        site_name = str(info.get("trainer_site") or "").strip()
        label = f"Train: {unit_name}"
        if xp_gained > 0:
            label += f" +{xp_gained} XP"
        if site_name:
            label += f" ({site_name})"
        return label

    try:
        action_idx = int(action)
    except Exception:
        return str(action)

    if 0 <= action_idx < len(GRID_MOVE_ACTION_NAMES):
        return GRID_MOVE_ACTION_NAMES[action_idx]
    return f"#{action_idx}"


# ======================= BATTLE VISUALIZER =======================

class BattleVisualizer:
    """Полная визуализация боя (портирована из visualize.py)."""

    # Константы позиций
    RED_FRONT = [1, 2, 3]
    RED_BACK = [4, 5, 6]
    BLUE_FRONT = [7, 8, 9]
    BLUE_BACK = [10, 11, 12]
    ALL_POS = RED_FRONT + RED_BACK + BLUE_FRONT + BLUE_BACK

    # Координаты
    COL_X = {0: 0.18, 1: 0.50, 2: 0.82}
    Y_BLUE_BACK, Y_BLUE_FRONT = 0.88, 0.70
    Y_RED_FRONT, Y_RED_BACK = 0.30, 0.12
    SLOT_W, SLOT_H = 0.31, 0.16
    HP_H = 0.028

    FRAME_DELAY = 0.14  # Базовая задержка между кадрами

    # Статы призванных юнитов
    OCCULTMASTER_SUMMON_STATS = {
        "Дракон смерти": ("Mage", 375, True, 0.0),
        "Драколич": ("Mage", 525, True, 0.0),
        "Вампир": ("Vampire", 185, False, 0.0),
        "Дух": ("Archer", 150, False, 0.0),
        "Лич": ("Mage", 140, False, 0.0),
        "Тень": ("Shadow", 135, False, 0.0),
        "Сущий": ("Wight", 105, False, 80.0),
        "Гигантский чёрный паук": ("Spider", 370, True, 90.0),
        "Чёрный дракон": ("Mage", 800, True, 0.0),
        "Высший вампир": ("Mage", 210, False, 0.0),
    }

    # Регулярные выражения для парсинга логов
    atk_re = re.compile(r"^(RED|BLUE)\s+[^#]+#(\d+)\s*[>→]\s*(RED|BLUE)\s+[^#]+#(\d+):\s+(\d+)\s+\((\d+)[>→](\d+)\)")
    kill_re = re.compile(r"^\?\s+(RED|BLUE)\s+[^#]+#(\d+)\s+выведен из строя\.")
    linked_summon_kill_re = re.compile(r"^\?\s+(RED|BLUE)\s+[^#]+#(\d+)\s+гибнет вместе с призывателем\s+[^#]+#(\d+)\s+\((\d+)>0\)\.")
    vict_re = re.compile(r"^\?\?\s*Победа\s+(RED|BLUE)!")
    blue_turn_re = re.compile(r"^Ход\s+BLUE:\s+[^#]+#(\d+)")
    red_turn_re = re.compile(r"^RED ход:\s+[^#]+#(\d+)")
    blue_action_re = re.compile(r"^BLUE действие:\s+[^#]+#(\d+)\s*[>→]\s*pos(\d+)")
    red_target_re = re.compile(r"^RED ход:\s+[^#]+#(\d+).+цель.*pos(\d+)")
    mage_banner_re = re.compile(r'^\w+\s+[^#]+#(\d+)\s+\((?:Mage|Dead dragon|Wolf Lord|Gumtic|Drulliaan|Uter Demon|"Shamanka"|Tiamat|Teurg)\).+массов', re.IGNORECASE)
    blue_cant_re = re.compile(r"^BLUE\s+[^#]+#(\d+).+не может достать pos(\d+)")
    poison_tick_re = re.compile(r"^\?\s*Яд поражает (RED|BLUE)\s+[^#]+#(\d+):\s+(\d+)\s+\((\d+)[>→](\d+)\)")
    burn_tick_re = re.compile(r"^\?\?\s*Поджог поражает (RED|BLUE)\s+[^#]+#(\d+):\s+(\d+)\s+\((\d+)[>→](\d+)\)")
    uran_tick_re = re.compile(r"^\?\s*Вода поражает (RED|BLUE)\s+[^#]+#(\d+):\s+(\d+)\s+\((\d+)[>→](\d+)\)")
    heal_re = re.compile(r"^\?\s*Лечение: (RED|BLUE)\s+[^#]+#(\d+)\s+восстанавливает\s+(\d+)\s+HP\s+для\s+(RED|BLUE)\s+[^#]+#(\d+)\s+\((\d+)[>→](\d+)\)\.")
    patriach_revive_re = re.compile(r"^\?\s*Воскрешение Патриарха:\s+(RED|BLUE)\s+[^#]+#(\d+)\s+возвращает\s+(RED|BLUE)\s+[^#]+#(\d+)\s+к жизни\s+\((\d+)\/(\d+)\)\.$")
    vamp_leech_re = re.compile(r"^\?\?\s*Кровопийство:\s+(RED|BLUE)\s+[^#]+#(\d+)\s+восстанавливает\s+(\d+)\s+HP\s+\((\d+)[>→](\d+)\)\s+из\s+(\d+)\s+(?:нанес[её]нных|похищ[её]нных)\.?$")
    blood_share_re = re.compile(r"^\?\?\s*Делёж крови:\s+(RED|BLUE)\s+[^#]+#(\d+)\s+(?:передаёт|добавляет)\s+(\d+)\s+HP\s+для\s+(RED|BLUE)\s+[^#]+#(\d+)\s+\((\d+)[>→](\d+)\)\.$")
    heal_mass_re = re.compile(r"^\?\s*Массовое лечение:\s+(RED|BLUE)\s+[^#]+#(\d+)\s+восстанавливает союзникам здоровье\.")
    immune_dmg_re = re.compile(r"^\?\?\s*Иммунитет к урону")
    immune_stat_re = re.compile(r"^\?\?\s*Иммунитет к эффекту")
    resist_re = re.compile(r"^\?\?\s*Стойкость")
    miss_re = re.compile(r"^\?\?\s*Промах:\s+(RED|BLUE)\s+[^#]+#(\d+)\s+по\s+(RED|BLUE)\s+[^#]+#(\d+)\.")
    paralysis_apply_re = re.compile(r"^Паралич:\s+(RED|BLUE)\s+[^#]+#(\d+)\s+лишает хода\s+(RED|BLUE)\s+[^#]+#(\d+)")
    paralysis_skip_re = re.compile(r"^\?\s*Паралич:\s+(RED|BLUE)\s+[^#]+#(\d+)\s+пропускает ход\.")
    long_paralysis_pass_re = re.compile(r"^\?\s*Пас:\s+(RED|BLUE)\s+[^#]+#(\d+)\s+остаётся парализован\.")
    long_paralysis_recover_re = re.compile(r"^\?\s*Долгий паралич:\s+(RED|BLUE)\s+[^#]+#(\d+)\s+пропускает ход и восстанавливается\.")
    long_paralysis_chance_re = re.compile(r"^\?\s*Шанс долгого паралича")
    long_paralysis_apply_re = re.compile(r"^\?\s*Долгий паралич:\s+(RED|BLUE)\s+[^#]+#(\d+)\s+накладывает долгий паралич на\s+(RED|BLUE)\s+[^#]+#(\d+)")
    runaway_re = re.compile(r"^\?\?\s+(RED|BLUE)\s+[^#]+#(\d+)\s+.*покидает бой\.")
    fear_re = re.compile(r"^\?\?\s*(RED|BLUE)\s+[^#]+#(\d+)\s+вселяет страх.*#(\d+)")
    summon_re = re.compile(r"^\?\?\s*Призыв:\s+(RED|BLUE)\s+[^#]+#(\d+)\s+призывает\s+(.+?)\s+на\s+pos(\d+)\s+\((ahead|behind)\)\.")
    doppel_pick_re = re.compile(r"^\?\?\s*Doppelganger выбор:\s*pos(\d+)\s*→\s*(RED|BLUE)\s+[^#]+#(\d+)\s*$")
    doppel_empty_re = re.compile(r"^\?\?\s*Doppelganger выбор:\s*pos(\d+)\s*→\s*пусто\s*$")
    wolf_fenrir_re = re.compile(r"^BLUE действие:\s+Повелитель волков#(\d+)\s*[>→]\s*pos\1\s*\(своя позиция выбрана\)")
    occ_summon_re = re.compile(r"^\?\? OCCULTMASTER: призван (?P<name>.+?) на pos(?P<pos>\d+)\s*\((?P<stand>ahead|behind)\)\.")
    laclaan_summon_re = re.compile(r"^\?\? LACLAAN: призван (?P<name>.+?) на pos(?P<pos>\d+)\s*\((?P<stand>ahead|behind)\)\.")
    centaur_bleed_re = re.compile(r"^\?\s*(RED|BLUE)\s+[^#]+#(\d+)\s+applies centaur bleed\s+\(([\d.]+)\)\s+to\s+(RED|BLUE)\s+[^#]+#(\d+)\s+\(([\d.]+)[>→]([\d.]+)\)\.?$")

    def __init__(self, fig=None, ax=None, speed_mult: float = 1.0):
        """Инициализация визуализатора боя."""
        self.state = {}
        self.active_pos = None
        self.handle = None
        # speed_mult > 1.0 ускоряет анимацию, < 1.0 замедляет.
        self.speed_mult = max(0.01, float(speed_mult))

        if fig is None or ax is None:
            self.fig, self.ax = plt.subplots(figsize=(13.5, 8.5))
        else:
            self.fig, self.ax = fig, ax

        # Пробуем использовать IPython display
        try:
            self.handle = display(self.fig, display_id=True)
            if self.handle is not None:
                plt.close(self.fig)
        except Exception:
            self.handle = None

        if self.handle is None:
            plt.ion()
            if plt.get_backend().lower() != "agg":
                try:
                    self.fig.show()
                except Exception:
                    pass

    def _suffix_by_type(self, t: str) -> str:
        return {
            "Mage": " (Mage)", "Wolf Lord": " (Wolf Lord)", "Warrior": " (Warrior)",
            "Centaur Savage": " (Centaur Savage)", "Doppelganger": " (Doppelganger)",
            "Demon": " (Demon)", "Death": " (Death)", "Wight": " (Wight)",
            "Archer": " (Archer)", "Lord": " (Lord)", "Bone Lord": " (Bone Lord)",
            "Dregazul": " (Dregazul)", "Dead dragon": " (Dead dragon)",
            "Ismir son": " (Ismir son)", "Ghost": " (Ghost)", "Shadow": " (Shadow)",
            "Succub": " (Succub)", "Betrezen": " (Betrezen)", "Uter": " (Uter)",
            "Uter Demon": " (Uter Demon)", "Tiamat": " (Tiamat)", "Sentry": " (Sentry)",
            "Watcher": " (Watcher)", "Teurg": " (Teurg)", "Incub": " (Incub)",
            "Abyss Devil": " (Abyss Devil)", "Sundancer": " (Sundancer)",
            "Sylfid": " (Sylfid)", "Deva roshi": " (Deva roshi)", "Patriach": " (Patriach)",
            "Novice": " (Novice)", "Travnitsa": " (Travnitsa)", "Dwarfdruid": " (Dwarfdruid)",
            "Arhidruid": " (Arhidruid)", "Spider": " (Spider)", "Drulliaan": " (Drulliaan)",
            "Elfarcher": " (Elfarcher)", "Shamanka": " (Shamanka)",
        }.get(t, f" ({t})")

    def init_state_from_units(self, units: list):
        """Инициализирует состояние визуализации из списка юнитов."""
        self.state = {}
        for u in units:
            t = u.get("unit_type", "Archer")
            start_hp = float(u.get("health", 0))
            max_hp = float(u.get("max_health", start_hp) or start_hp)
            self.state[u["position"]] = {
                "team": u["team"],
                "name": (u.get("name") or f"unit{u['position']}") + self._suffix_by_type(t),
                "hp": start_hp, "maxhp": max_hp, "stand": u.get("stand", "ahead"), "type": t,
                "big": bool(u.get("big", False)),
                "acc2": float(u.get("accuracy_secondary", 0)),
                "paralyzed": int(u.get("paralyzed", 0)),
                "running_away": int(u.get("running_away", 0)),
                "escaped": False,
            }

    def sync_hp_from_units(self, units: list):
        """Синхронизирует HP всех юнитов из battle_env.combined."""
        for u in units:
            pos = u["position"]
            if pos in self.state:
                new_hp = float(u.get("health", 0))
                new_max = float(u.get("max_health", new_hp) or new_hp)
                self.state[pos]["hp"] = new_hp
                self.state[pos]["maxhp"] = new_max
                self.state[pos]["paralyzed"] = int(u.get("paralyzed", 0))
                self.state[pos]["running_away"] = int(u.get("running_away", 0))

    def pos_to_xy(self, pos: int):
        col = (pos - 1) % 3
        x = self.COL_X[col] - self.SLOT_W / 2
        if pos in self.BLUE_BACK:
            y = self.Y_BLUE_BACK - self.SLOT_H / 2
        elif pos in self.BLUE_FRONT:
            y = self.Y_BLUE_FRONT - self.SLOT_H / 2
        elif pos in self.RED_FRONT:
            y = self.Y_RED_FRONT - self.SLOT_H / 2
        else:
            y = self.Y_RED_BACK - self.SLOT_H / 2
        return x, y

    def pos_center(self, pos: int):
        x, y = self.pos_to_xy(pos)
        return x + self.SLOT_W / 2, y + self.SLOT_H / 2

    def _set_hp(self, pos: int, hp_value: float):
        clamped = max(0.0, hp_value)
        if pos in self.state:
            self.state[pos]["hp"] = clamped
        partner = None
        if pos in self.state and self.state[pos].get("big", False):
            partner = pos + 3
        elif (pos - 3) in self.state and self.state[pos - 3].get("big", False):
            partner = pos - 3
        if partner in self.state:
            self.state[partner]["hp"] = clamped

    def _set_paralysis(self, pos: int, value: int):
        if pos in self.state:
            self.state[pos]["paralyzed"] = value
        partner = None
        if pos in self.state and self.state[pos].get("big", False):
            partner = pos + 3
        elif (pos - 3) in self.state and self.state[pos - 3].get("big", False):
            partner = pos - 3
        if partner in self.state:
            self.state[partner]["paralyzed"] = value

    def _spawn_visual_entry(self, dest_pos, team, unit_name, unit_type="Warrior", stand="ahead", hp=170.0, maxhp=170.0, big=False, acc2=0.0):
        team_norm = team.lower()
        self.state[dest_pos] = {
            "team": team_norm,
            "name": f"{unit_name}{self._suffix_by_type(unit_type)}",
            "hp": float(hp), "maxhp": float(maxhp), "stand": stand,
            "type": unit_type, "big": bool(big), "acc2": float(acc2),
            "paralyzed": 0, "running_away": 0, "escaped": False,
        }
        if big:
            partner = dest_pos + 3 if dest_pos in (self.RED_FRONT + self.BLUE_FRONT) else dest_pos - 3
            if partner in self.state:
                self.state[partner] = dict(self.state[dest_pos])
                self.state[partner]["name"] = self.state[dest_pos]["name"]
                self.state[partner]["stand"] = "behind" if self.state[dest_pos]["stand"] == "ahead" else "ahead"

    def _strip_suffix(self, name: str) -> str:
        return re.sub(r"\s*\([^)]*\)\s*$", "", name or "").strip()

    def _wrap_text(self, text: str, width: int, max_lines: int | None = None) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        lines = textwrap.wrap(
            raw,
            width=max(1, int(width)),
            break_long_words=False,
            break_on_hyphens=False,
        ) or [raw]
        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]
        return "\n".join(lines)

    def _doppel_morph(self, attacker_pos: int, target_pos: int):
        if attacker_pos not in self.state or target_pos not in self.state:
            return
        src = self.state[attacker_pos]
        dst = self.state[target_pos]
        ratio = 0.0
        if src["maxhp"] > 0:
            ratio = max(0.0, min(1.0, src["hp"] / src["maxhp"]))
        new_type = dst["type"]
        src["type"] = new_type
        src["name"] = self._strip_suffix(src["name"]) + self._suffix_by_type(new_type)
        src["maxhp"] = float(dst["maxhp"])
        src["hp"] = float(int(round(dst["hp"] * ratio)))
        src["acc2"] = float(dst.get("acc2", 0.0))
        src["big"] = False
        src["paralyzed"] = 0
        src["running_away"] = 0

    def _wolflord_to_fenrir(self, pos: int):
        if pos not in self.state:
            return
        u = self.state[pos]
        if self._strip_suffix(u.get("name", "")) != "Повелитель волков":
            return
        old_hp = float(u.get("hp", 0.0))
        old_max = float(u.get("maxhp", old_hp if old_hp > 0 else 1.0))
        ratio = 0.0 if old_max <= 0 else max(0.0, min(1.0, old_hp / old_max))
        u["type"] = "Warrior"
        u["maxhp"] = 275.0
        u["hp"] = float(int(round(275.0 * ratio)))
        u["name"] = "Дух Фенрира" + self._suffix_by_type(u["type"])

    def draw_unit(self, ax, pos, is_active: bool = False):
        if pos not in self.state:
            return
        u = self.state[pos]
        x, y = self.pos_to_xy(pos)
        if pos in (self.RED_BACK + self.BLUE_BACK) and u["hp"] <= 0:
            return
        card_color = (0.90, 0.30, 0.30, 0.16) if u["team"] == "red" else (0.30, 0.45, 0.90, 0.16)
        ax.add_patch(patches.FancyBboxPatch((x, y), self.SLOT_W, self.SLOT_H, boxstyle="round,pad=0.010,rounding_size=0.016", linewidth=1.3, edgecolor="black", facecolor=card_color))
        if is_active:
            ax.add_patch(patches.FancyBboxPatch((x - 0.01, y - 0.01), self.SLOT_W + 0.02, self.SLOT_H + 0.02, boxstyle="round,pad=0.012,rounding_size=0.018", linewidth=3.0, edgecolor=(1.0, 0.82, 0.10), facecolor="none", alpha=0.95))
        label_text = self._wrap_text(u["name"], width=24, max_lines=2)
        ax.text(
            x + 0.014,
            y + self.SLOT_H * 0.86,
            label_text,
            fontsize=9.2,
            fontweight="bold",
            ha="left",
            va="top",
            color="black",
            linespacing=1.0,
        )
        if u.get("acc2", 0) > 0:
            ax.text(x + self.SLOT_W - 0.014, y + self.SLOT_H * 0.86, f"{int(u['acc2'])}%", fontsize=9, ha="right", va="top", color="black")
        hp_x, hp_y = x + 0.014, y + self.SLOT_H * 0.08
        hp_w = self.SLOT_W - 0.028
        ax.add_patch(patches.Rectangle((hp_x, hp_y), hp_w, self.HP_H, facecolor=(0.88, 0.88, 0.88), edgecolor="none"))
        frac = max(0.0, min(1.0, u["hp"] / max(1e-9, u["maxhp"])))
        ax.add_patch(patches.Rectangle((hp_x, hp_y), hp_w * frac, self.HP_H, facecolor=(0.15, 0.70, 0.25), edgecolor="none"))
        ax.text(hp_x + hp_w / 2, hp_y + self.HP_H / 2, f"{int(max(0, u['hp']))}/{int(u['maxhp'])}", fontsize=9, ha="center", va="center", color="black")
        if u.get("paralyzed", 0):
            ax.text(x + self.SLOT_W / 2, y + self.SLOT_H * 0.38, "PAR", fontsize=9, fontweight="bold", ha="center", va="center", color=(0.75, 0.05, 0.05))
        if u.get("running_away", 0):
            ax.add_patch(patches.Rectangle((x + self.SLOT_W * 0.08, y + self.SLOT_H * 0.30), self.SLOT_W * 0.84, self.SLOT_H * 0.18, facecolor=(0.80, 0.65, 0.10, 0.35), edgecolor="none"))
            ax.text(x + self.SLOT_W / 2, y + self.SLOT_H * 0.39, "RUN", fontsize=9, fontweight="bold", ha="center", va="center", color=(0.6, 0.15, 0.05))
        if u.get("escaped"):
            ax.text(x + self.SLOT_W / 2, y + self.SLOT_H * 0.62, "ESCAPED", fontsize=10, fontweight="bold", ha="center", va="center", color=(0.5, 0.1, 0.1))
        ax.text(x + self.SLOT_W / 2, y - 0.008, f"pos{pos}", fontsize=8, ha="center", va="top", color=(0.2, 0.2, 0.2))

    def draw_big_unit(self, ax, front_pos: int, is_active: bool = False):
        back_pos = front_pos + 3
        if front_pos not in self.state:
            return
        uf = self.state[front_pos]
        if uf["hp"] <= 0:
            return
        x_front, y_front = self.pos_to_xy(front_pos)
        x_back, y_back = self.pos_to_xy(back_pos)
        x = x_front
        y = min(y_front, y_back)
        height = (max(y_front, y_back) + self.SLOT_H) - y
        width = self.SLOT_W
        card_color = (0.90, 0.30, 0.30, 0.20) if uf["team"] == "red" else (0.30, 0.45, 0.90, 0.20)
        ax.add_patch(patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.020", linewidth=1.6, edgecolor="black", facecolor=card_color))
        if is_active:
            ax.add_patch(patches.FancyBboxPatch((x - 0.012, y - 0.012), width + 0.024, height + 0.024, boxstyle="round,pad=0.012,rounding_size=0.022", linewidth=3.2, edgecolor=(1.0, 0.82, 0.10), facecolor="none", alpha=0.95))
        label_text = self._wrap_text(uf["name"], width=24, max_lines=3)
        ax.text(
            x + 0.014,
            y + height * 0.90,
            label_text,
            fontsize=9.8,
            fontweight="bold",
            ha="left",
            va="top",
            color="black",
            linespacing=1.0,
        )
        if uf.get("acc2", 0) > 0:
            ax.text(x + width - 0.014, y + height * 0.90, f"{int(uf['acc2'])}%", fontsize=10, ha="right", va="top", color="black")
        hp_w = width - 0.028
        hp_x, hp_y = x + 0.014, y + height * 0.08
        ax.add_patch(patches.Rectangle((hp_x, hp_y), hp_w, self.HP_H, facecolor=(0.88, 0.88, 0.88), edgecolor="none"))
        frac = max(0.0, min(1.0, uf["hp"] / max(1e-9, uf["maxhp"])))
        ax.add_patch(patches.Rectangle((hp_x, hp_y), hp_w * frac, self.HP_H, facecolor=(0.15, 0.70, 0.25), edgecolor="none"))
        ax.text(hp_x + hp_w / 2, hp_y + self.HP_H / 2, f"{int(max(0, uf['hp']))}/{int(uf['maxhp'])}", fontsize=9, ha="center", va="center", color="black")
        ax.text(x + width / 2, y - 0.010, f"pos{front_pos}+{back_pos}", fontsize=8, ha="center", va="top", color=(0.2, 0.2, 0.2))
        if uf.get("paralyzed", 0):
            ax.text(x + width / 2, y + height * 0.46, "PAR", fontsize=10, fontweight="bold", ha="center", va="center", color=(0.75, 0.05, 0.05))

    def draw_attack_arrow(self, ax, src_pos, dst_pos, team, text=None, style="solid", color=None, alpha=0.95, curve=0.18, lw=2.6):
        sx, sy = self.pos_center(src_pos)
        dx, dy = self.pos_center(dst_pos)
        vx, vy = dx - sx, dy - sy
        dist = math.hypot(vx, vy) + 1e-9
        pad = 0.06
        sx += vx / dist * pad
        sy += vy / dist * pad
        dx -= vx / dist * pad
        dy -= vy / dist * pad
        if color is None:
            color = (0.20, 0.35, 0.85) if team.upper() == "BLUE" else (0.85, 0.25, 0.25)
        con_style = f"arc3,rad={curve}" if abs(vx) > 0.01 and abs(vy) > 0.01 else "arc3,rad=0.0"
        arrow = FancyArrowPatch((sx, sy), (dx, dy), arrowstyle="-|>", mutation_scale=14, linewidth=lw, linestyle="--" if style == "dashed" else "solid", color=color, alpha=alpha, connectionstyle=con_style)
        ax.add_patch(arrow)
        if text:
            mx, my = (sx + dx) / 2, (sy + dy) / 2
            ax.text(mx, my + 0.03, text, fontsize=10, fontweight="bold", ha="center", va="center", color=color)

    def draw_board(self, arrows=None, headline="", active_pos=None):
        self.ax.cla()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        self.ax.plot([0.02, 0.98], [0.50, 0.50], color=(0.6, 0.6, 0.6), lw=1.2, ls="--", alpha=0.7)
        self.ax.text(0.01, 0.96, "BLUE (top)", color=(0.2, 0.35, 0.8), fontsize=12, fontweight="bold", ha="left")
        self.ax.text(0.01, 0.04, "RED (bottom)", color=(0.8, 0.25, 0.25), fontsize=12, fontweight="bold", ha="left")
        arrows = arrows or []
        drawn = set()
        for front_pos in self.RED_FRONT + self.BLUE_FRONT:
            if front_pos in self.state and self.state[front_pos].get("big", False):
                self.draw_big_unit(self.ax, front_pos, is_active=(active_pos in (front_pos, front_pos + 3)))
                drawn.add(front_pos)
                drawn.add(front_pos + 3)
        for pos in self.ALL_POS:
            if pos in drawn:
                continue
            self.draw_unit(self.ax, pos, is_active=(active_pos == pos))
        for a in arrows:
            try:
                self.draw_attack_arrow(self.ax, a["src"], a["dst"], a.get("team", "RED"), text=a.get("text"), style=a.get("style", "solid"), color=a.get("color"), alpha=a.get("alpha", 0.95), curve=a.get("curve", 0.18), lw=a.get("lw", 2.6))
            except Exception:
                pass
        if headline:
            headline_text = self._wrap_text(headline, width=72, max_lines=3)
            self.ax.text(
                0.5,
                0.99,
                headline_text,
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="top",
                color="black",
                linespacing=1.05,
                bbox=dict(boxstyle="round,pad=0.30", facecolor=(1.0, 1.0, 1.0, 0.82), edgecolor="none"),
            )
        if self.handle is not None:
            self.handle.update(self.fig)
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def wait(self, frames=1.0):
        time.sleep((self.FRAME_DELAY * frames) / self.speed_mult)

    def _render_state(self):
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        drawn_big_front = set()
        for pos in self.ALL_POS:
            u = self.state.get(pos)
            if not u or u.get("hp", 0) <= 0:
                continue
            if u.get("big") and pos in (self.RED_FRONT + self.BLUE_FRONT):
                if pos not in drawn_big_front:
                    self.draw_big_unit(self.ax, pos)
                    drawn_big_front.add(pos)
            elif not (u.get("big") and pos in (self.RED_BACK + self.BLUE_BACK)):
                self.draw_unit(self.ax, pos)
        if self.handle is not None:
            self.handle.update(self.fig)
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def process_log_line(self, line: str):
        """Обрабатывает одну строку лога и обновляет визуализацию."""
        line = line.strip()
        if not line:
            return

        m = self.blue_turn_re.match(line)
        if m:
            self.active_pos = int(m.group(1))
            self.draw_board(headline=line, active_pos=self.active_pos)
            self.wait(0.9)
            return

        m = self.red_turn_re.match(line)
        if m:
            self.active_pos = int(m.group(1))
            self.draw_board(headline=line, active_pos=self.active_pos)
            self.wait(0.9)
            return

        m = self.wolf_fenrir_re.match(line)
        if m:
            pos = int(m.group(1))
            self._wolflord_to_fenrir(pos)
            self.draw_board(arrows=[{"src": pos, "dst": pos, "team": "BLUE", "style": "dashed", "text": "FENRIR", "color": (0.20, 0.35, 0.85), "lw": 2.6, "alpha": 0.95}], headline=line, active_pos=pos)
            self.wait(0.8)
            return

        m = self.blue_action_re.match(line)
        if m:
            src, dst = int(m.group(1)), int(m.group(2))
            self.draw_board(arrows=[{"src": src, "dst": dst, "team": "BLUE", "style": "dashed", "alpha": 0.6}], headline=line, active_pos=src)
            self.wait(0.6)
            return

        m = self.red_target_re.match(line)
        if m:
            src, dst = int(m.group(1)), int(m.group(2))
            self.draw_board(arrows=[{"src": src, "dst": dst, "team": "RED", "style": "dashed", "alpha": 0.6}], headline=line, active_pos=src)
            self.wait(0.6)
            return

        if self.mage_banner_re.match(line):
            self.draw_board(headline=line + " (AOE)", active_pos=int(self.mage_banner_re.match(line).group(1)))
            self.wait(0.6)
            return

        m = self.summon_re.match(line)
        if m:
            team, src_pos, unit_name, dest_pos, stand = m.group(1), int(m.group(2)), m.group(3), int(m.group(4)), m.group(5)
            norm_name = (unit_name or "").strip().lower()
            summon_stats = {"unit_type": "Warrior", "hp": 170.0, "maxhp": 170.0}
            if "зомби" in norm_name:
                summon_stats.update({"unit_type": "Warrior", "hp": 170.0, "maxhp": 170.0})
            elif "элементаль воздуха" in norm_name:
                summon_stats.update({"unit_type": "Archer", "hp": 100.0, "maxhp": 100.0})
            elif "малый энт" in norm_name:
                summon_stats.update({"unit_type": "Warrior", "hp": 80.0, "maxhp": 80.0})
            self._spawn_visual_entry(dest_pos, team, unit_name, unit_type=summon_stats["unit_type"], stand=stand, hp=summon_stats["hp"], maxhp=summon_stats["maxhp"])
            self.draw_board(arrows=[{"src": src_pos, "dst": dest_pos, "team": team, "style": "dashed", "text": "SUMMON"}], headline=line, active_pos=src_pos)
            self.wait(0.8)
            return

        m = self.occ_summon_re.match(line)
        if m:
            name, pos, stand = m.group("name").strip(), int(m.group("pos")), m.group("stand")
            team = "red" if pos in (self.RED_FRONT + self.RED_BACK) else "blue"
            unit_type, maxhp, is_big, acc2 = self.OCCULTMASTER_SUMMON_STATS.get(name, ("Warrior", 150.0, False, 0.0))
            self._spawn_visual_entry(pos, team, name, unit_type=unit_type, stand=stand, hp=maxhp, maxhp=maxhp, big=is_big, acc2=acc2)
            return

        m = self.laclaan_summon_re.match(line)
        if m:
            name, pos, stand = m.group("name").strip(), int(m.group("pos")), m.group("stand")
            team = "red" if pos in (self.RED_FRONT + self.RED_BACK) else "blue"
            unit_type, maxhp, is_big, acc2 = self.OCCULTMASTER_SUMMON_STATS.get(name, ("Warrior", 150.0, False, 0.0))
            self._spawn_visual_entry(pos, team, name, unit_type=unit_type, stand=stand, hp=maxhp, maxhp=maxhp, big=is_big, acc2=acc2)
            return

        m = self.doppel_pick_re.match(line)
        if m:
            self.active_pos = int(m.group(1))
            target_pos = int(m.group(3))
            self._doppel_morph(self.active_pos, target_pos)
            self.draw_board(arrows=[{"src": self.active_pos, "dst": target_pos, "team": "BLUE", "style": "dashed", "text": "MORPH", "color": (0.20, 0.35, 0.85)}], headline=line, active_pos=self.active_pos)
            self.wait(0.8)
            return

        if self.doppel_empty_re.match(line):
            self.draw_board(headline=line, active_pos=self.active_pos)
            self.wait(0.5)
            return

        m = self.atk_re.match(line)
        if m:
            src_team = m.group(1)
            src_pos = int(m.group(2))
            dst_pos = int(m.group(4))
            dmg = int(m.group(5))
            after = int(m.group(7))
            self._set_hp(dst_pos, after)
            self.draw_board(arrows=[{"src": src_pos, "dst": dst_pos, "team": src_team, "text": str(dmg)}], headline=line, active_pos=src_pos)
            self.wait(0.9)
            return

        m = self.miss_re.match(line)
        if m:
            src_team, src_pos, dst_pos = m.group(1), int(m.group(2)), int(m.group(4))
            self.draw_board(arrows=[{"src": src_pos, "dst": dst_pos, "team": src_team, "style": "dashed"}], headline=line, active_pos=src_pos)
            self.wait(0.6)
            return

        if self.immune_dmg_re.match(line) or self.immune_stat_re.match(line) or self.resist_re.match(line):
            self.draw_board(headline=line, active_pos=self.active_pos)
            self.wait(0.5)
            return

        m = self.poison_tick_re.match(line)
        if m:
            self._set_hp(int(m.group(2)), int(m.group(5)))
            self.draw_board(headline=line, active_pos=int(m.group(2)))
            self.wait(0.7)
            return

        m = self.burn_tick_re.match(line)
        if m:
            self._set_hp(int(m.group(2)), int(m.group(5)))
            self.draw_board(headline=line, active_pos=int(m.group(2)))
            self.wait(0.7)
            return

        m = self.uran_tick_re.match(line)
        if m:
            self._set_hp(int(m.group(2)), int(m.group(5)))
            self.draw_board(headline=line, active_pos=int(m.group(2)))
            self.wait(0.7)
            return

        m = self.patriach_revive_re.match(line)
        if m:
            healer_pos, rec_pos, restored, maxhp = int(m.group(2)), int(m.group(4)), int(m.group(5)), int(m.group(6))
            self._set_hp(rec_pos, restored)
            self.state[rec_pos]["maxhp"] = float(maxhp)
            self.draw_board(arrows=[{"src": healer_pos, "dst": rec_pos, "team": m.group(1), "text": f"REVIVE {restored}/{maxhp}", "color": (0.95, 0.82, 0.15), "lw": 2.6}], headline=line, active_pos=healer_pos)
            self.wait(0.9)
            return

        m = self.heal_re.match(line)
        if m:
            healer_pos, rec_pos, healed, after = int(m.group(2)), int(m.group(5)), int(m.group(3)), int(m.group(7))
            self._set_hp(rec_pos, after)
            self.draw_board(arrows=[{"src": healer_pos, "dst": rec_pos, "team": m.group(1), "text": f"+{healed}", "color": (0.15, 0.65, 0.25), "lw": 2.2}], headline=line, active_pos=healer_pos)
            self.wait(0.8)
            return

        m = self.vamp_leech_re.match(line)
        if m:
            pos, healed, after = int(m.group(2)), int(m.group(3)), int(m.group(5))
            self._set_hp(pos, after)
            self.draw_board(arrows=[{"src": pos, "dst": pos, "team": m.group(1), "text": f"+{healed}", "color": (0.15, 0.70, 0.30), "lw": 2.0}], headline=line, active_pos=pos)
            self.wait(0.8)
            return

        m = self.blood_share_re.match(line)
        if m:
            donor_pos, rec_pos, healed, after = int(m.group(2)), int(m.group(5)), int(m.group(3)), int(m.group(7))
            self._set_hp(rec_pos, after)
            self.draw_board(arrows=[{"src": donor_pos, "dst": rec_pos, "team": m.group(1), "text": f"+{healed}", "color": (0.15, 0.70, 0.30), "lw": 2.2}], headline=line, active_pos=donor_pos)
            self.wait(0.8)
            return

        if self.heal_mass_re.match(line):
            self.draw_board(headline=line, active_pos=self.active_pos)
            self.wait(0.7)
            return

        m = self.paralysis_apply_re.match(line)
        if m:
            src_pos, dst_pos = int(m.group(2)), int(m.group(4))
            self._set_paralysis(dst_pos, 1)
            self.draw_board(arrows=[{"src": src_pos, "dst": dst_pos, "team": m.group(1), "style": "solid", "color": (0.55, 0.2, 0.7)}], headline=line, active_pos=src_pos)
            self.wait(0.8)
            return

        if self.long_paralysis_apply_re.match(line) or self.long_paralysis_chance_re.match(line):
            self.draw_board(headline=line, active_pos=self.active_pos)
            self.wait(0.6)
            return

        if self.paralysis_skip_re.match(line):
            self._set_paralysis(int(self.paralysis_skip_re.match(line).group(2)), 0)
            self.draw_board(headline=line, active_pos=int(self.paralysis_skip_re.match(line).group(2)))
            self.wait(0.6)
            return

        if self.long_paralysis_pass_re.match(line):
            self.draw_board(headline=line, active_pos=self.active_pos)
            self.wait(0.6)
            return

        if self.long_paralysis_recover_re.match(line):
            self._set_paralysis(int(self.long_paralysis_recover_re.match(line).group(2)), 0)
            self.draw_board(headline=line, active_pos=int(self.long_paralysis_recover_re.match(line).group(2)))
            self.wait(0.6)
            return

        m = self.fear_re.match(line)
        if m:
            src_team, src_pos, dst_pos = m.group(1), int(m.group(2)), int(m.group(3))
            if dst_pos in self.state:
                self.state[dst_pos]["running_away"] = 1
            self.draw_board(arrows=[{"src": src_pos, "dst": dst_pos, "team": src_team, "style": "dashed", "color": (0.6, 0.4, 0.0)}], headline=line, active_pos=src_pos)
            self.wait(0.7)
            return

        m = self.runaway_re.match(line)
        if m:
            pos = int(m.group(2))
            if pos in self.state:
                self.state[pos]["escaped"] = True
                self._set_hp(pos, 0)
                self.state[pos]["running_away"] = 0
                self._set_paralysis(pos, 0)
            self.draw_board(headline=line, active_pos=self.active_pos)
            self.wait(0.7)
            return

        m = self.blue_cant_re.match(line)
        if m:
            src, dst = int(m.group(1)), int(m.group(2))
            self.draw_board(arrows=[{"src": src, "dst": dst, "team": "BLUE", "style": "dashed", "alpha": 0.5}], headline=line, active_pos=src)
            self.wait(0.7)
            return

        m = self.kill_re.match(line)
        if m:
            pos = int(m.group(2))
            if pos in self.state:
                start_hp = float(self.state[pos].get("hp", 0))
                for v in np.linspace(start_hp, 0.0, 6):
                    self._set_hp(pos, v)
                    self._render_state()
                    time.sleep(self.FRAME_DELAY / self.speed_mult)
            return

        m = self.linked_summon_kill_re.match(line)
        if m:
            pos = int(m.group(2))
            start_hp = float(self.state.get(pos, {}).get("hp", float(m.group(4))))
            for v in np.linspace(start_hp, 0.0, 6):
                self._set_hp(pos, v)
                self._render_state()
                time.sleep(self.FRAME_DELAY / self.speed_mult)
            self._set_hp(pos, 0.0)
            return

        m = self.centaur_bleed_re.match(line)
        if m:
            src_team, src_pos, dmg, dst_pos, after = m.group(1), int(m.group(2)), float(m.group(3)), int(m.group(5)), float(m.group(7))
            self._set_hp(dst_pos, after)
            self.draw_board(arrows=[{"src": src_pos, "dst": dst_pos, "team": src_team, "text": f"{dmg:g}", "style": "solid", "color": (0.95, 0.55, 0.15), "lw": 3.0, "alpha": 0.98}], headline=line, active_pos=src_pos)
            self.wait(0.9)
            return

        m = self.vict_re.match(line)
        if m:
            self.draw_board(headline=f"{line}", active_pos=None)
            self.wait(1.2)
            return

        # Дефолтная отрисовка
        self.draw_board(headline=line, active_pos=self.active_pos)
        self.wait(0.35)

    def close(self):
        if self.handle is None:
            plt.ioff()
            try:
                plt.close(self.fig)
            except Exception:
                pass


# ======================= CAMPAIGN VISUALIZER =======================

class CampaignVisualizer:
    """Визуализатор для Campaign Mode."""

    # Цвета
    COLOR_AGENT = (0.2, 0.5, 0.9, 0.9)
    COLOR_SCRIPTED_BOT = (0.6, 0.15, 0.85, 0.95)
    COLOR_ENEMY_ALIVE = (0.9, 0.3, 0.3, 0.8)
    COLOR_ENEMY_DEAD = (0.5, 0.5, 0.5, 0.4)
    COLOR_DRAGON_TILE = (0.10, 0.75, 0.32, 0.28)
    COLOR_DRAGON_ALIVE = (0.10, 0.68, 0.22, 0.95)
    COLOR_DRAGON_DEAD = (0.35, 0.50, 0.35, 0.55)
    COLOR_VISITED = (0.7, 0.85, 0.7, 0.3)
    COLOR_UNVISITED = (0.95, 0.95, 0.95, 1.0)
    COLOR_PATH = (0.3, 0.6, 0.9, 0.6)
    COLOR_CASTLE = (1.0, 0.92, 0.2, 0.55)
    COLOR_LEGIONS_TERRITORY = (0.78, 0.16, 0.12, 0.14)
    COLOR_EMPIRE_TERRITORY = (0.18, 0.52, 0.20, 0.14)
    COLOR_OBSTACLE = (0.55, 0.55, 0.55, 0.9)
    COLOR_CHEST_FILL = (0.62, 0.40, 0.12, 0.95)
    COLOR_CHEST_EDGE = (0.30, 0.16, 0.02, 1.0)
    COLOR_MERCHANT_FILL = (0.96, 0.74, 0.16, 0.10)
    COLOR_MERCHANT_EDGE = (0.55, 0.28, 0.02, 0.32)
    COLOR_MERCHANT_SITE_FILL = (0.92, 0.18, 0.12, 0.06)
    COLOR_MERCHANT_SITE_EDGE = (0.88, 0.15, 0.08, 0.95)
    COLOR_MERCHANT_ANCHOR_FILL = (0.88, 0.10, 0.08, 0.34)
    COLOR_MERCHANT_ANCHOR_EDGE = (0.82, 0.05, 0.04, 0.98)
    COLOR_SPELL_SHOP_FILL = (0.14, 0.50, 0.86, 0.12)
    COLOR_SPELL_SHOP_EDGE = (0.06, 0.28, 0.58, 0.42)
    COLOR_SPELL_SHOP_ANCHOR_FILL = (0.08, 0.36, 0.72, 0.36)
    COLOR_SPELL_SHOP_ANCHOR_EDGE = (0.04, 0.20, 0.50, 0.98)
    COLOR_MERCENARY_FILL = (0.52, 0.18, 0.78, 0.12)
    COLOR_MERCENARY_EDGE = (0.34, 0.08, 0.56, 0.42)
    COLOR_MERCENARY_ANCHOR_FILL = (0.42, 0.10, 0.66, 0.36)
    COLOR_MERCENARY_ANCHOR_EDGE = (0.27, 0.04, 0.46, 0.98)
    COLOR_TRAINER_FILL = (0.96, 0.52, 0.12, 0.12)
    COLOR_TRAINER_EDGE = (0.70, 0.30, 0.02, 0.42)
    COLOR_TRAINER_ANCHOR_FILL = (0.92, 0.42, 0.06, 0.34)
    COLOR_TRAINER_ANCHOR_EDGE = (0.74, 0.23, 0.02, 0.98)
    COLOR_CAPITAL_FILL = (0.08, 0.30, 0.82, 0.18)
    COLOR_CAPITAL_EDGE = (0.05, 0.20, 0.62, 0.98)
    COLOR_VILLAGE_FILL = (0.00, 0.72, 0.78, 0.18)
    COLOR_VILLAGE_EDGE = (0.00, 0.50, 0.58, 0.98)
    COLOR_SETTLEMENT_CORNER_FILL = (1.0, 0.25, 0.62, 0.82)
    COLOR_SETTLEMENT_CORNER_EDGE = (0.58, 0.05, 0.30, 0.98)
    COLOR_RUIN_EDGE = (0.82, 0.05, 0.05, 0.98)
    COLOR_GOLD_MINE_FILL = (0.96, 0.80, 0.16, 0.84)
    COLOR_GOLD_MINE_EDGE = (0.56, 0.38, 0.02, 0.98)
    COLOR_GOLD_MINE_TEXT = (0.34, 0.20, 0.00, 0.98)
    CASTLE_POS = DEFAULT_HERO_GRID_POSITION
    INFO_PANEL_WIDTH_RATIO = 0.34

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        cell_size: float = 0.8,
        background_map_path: str | None = None,
        scenario_path: str | None = None,
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.path_history = []
        self.background_map_path = _existing_path_or_none(background_map_path)
        self.scenario_path = _existing_path_or_none(scenario_path)
        self.background_image = None
        self.capital_overlays: list[dict] = []
        self.village_overlays: list[dict] = []
        self.ruin_overlays: list[dict] = []
        self.gold_mine_overlays: list[dict] = []
        self._load_background_image()
        self._load_settlement_overlays()
        self._load_ruin_overlays()
        self._load_gold_mine_overlays()

        # Создаём фигуру
        self.fig = plt.figure(figsize=(15.5, 10))
        gs = self.fig.add_gridspec(
            1,
            2,
            width_ratios=[1.0, self.INFO_PANEL_WIDTH_RATIO],
            wspace=0.04,
        )
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.info_ax = self.fig.add_subplot(gs[0, 1])
        self.info_ax.axis("off")
        plt.ion()
        if plt.get_backend().lower() != "agg":
            self.fig.show()

    def _format_info_text(self, lines: list[str], width: int = 34) -> str:
        wrapped_lines: list[str] = []
        for raw_line in lines:
            line = str(raw_line or "")
            if not line:
                wrapped_lines.append("")
                continue
            initial_indent = ""
            subsequent_indent = ""
            content = line
            if line.startswith("- "):
                initial_indent = "- "
                subsequent_indent = "  "
                content = line[2:]
            wrapped = textwrap.wrap(
                content,
                width=max(8, int(width)),
                initial_indent=initial_indent,
                subsequent_indent=subsequent_indent,
                break_long_words=False,
                break_on_hyphens=False,
            )
            wrapped_lines.extend(wrapped or [line])
        return "\n".join(wrapped_lines)

    def _draw_info_panel(
        self,
        title: str,
        info_text: str,
        *,
        highlight_line: str | None = None,
    ) -> None:
        self.info_ax.clear()
        self.info_ax.set_facecolor((0.96, 0.97, 0.99))
        self.info_ax.axis("off")
        line_count = max(1, info_text.count("\n") + 1)
        if highlight_line:
            line_count += 1
        if line_count <= 34:
            body_fontsize = 10.5
        elif line_count <= 40:
            body_fontsize = 9.8
        elif line_count <= 48:
            body_fontsize = 9.1
        else:
            body_fontsize = 8.4
        panel = patches.FancyBboxPatch(
            (0.02, 0.02),
            0.96,
            0.96,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=1.3,
            edgecolor=(0.74, 0.79, 0.88),
            facecolor=(0.985, 0.99, 1.0),
            transform=self.info_ax.transAxes,
        )
        self.info_ax.add_patch(panel)
        if title:
            self.info_ax.text(
                0.06,
                0.965,
                self._format_info_text([title], width=30),
                transform=self.info_ax.transAxes,
                ha="left",
                va="top",
                fontsize=12.5,
                fontweight="bold",
                color="black",
                linespacing=1.1,
            )
        body_top = 0.92
        if highlight_line:
            r, g, b, _a = self.COLOR_SCRIPTED_BOT
            self.info_ax.text(
                0.06,
                body_top,
                str(highlight_line),
                transform=self.info_ax.transAxes,
                ha="left",
                va="top",
                fontsize=11.6,
                fontweight="bold",
                color=(r, g, b),
                linespacing=1.1,
            )
            body_top = 0.885
        self.info_ax.text(
            0.06,
            body_top,
            info_text,
            transform=self.info_ax.transAxes,
            ha="left",
            va="top",
            fontsize=body_fontsize,
            fontweight="bold",
            color="black",
            linespacing=1.15,
        )

    def _load_background_image(self) -> None:
        if not self.background_map_path:
            return
        try:
            image = plt.imread(self.background_map_path)
        except Exception as exc:
            print(f"[WARN] Failed to load background map {self.background_map_path}: {exc}")
            return

        if image.ndim < 2:
            return

        height = int(image.shape[0])
        width = int(image.shape[1])
        if width > height:
            image = image[:, :height].copy()
        elif height > width:
            image = image[:width, :, :].copy()

        self.background_image = self._sanitize_background_image(image)

    def _display_tile(self, x: int, y: int) -> tuple[int, int]:
        return self.grid_size - 1 - int(y), int(x)

    def _display_rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        return (
            self.grid_size - int(y) - int(height),
            int(x),
            int(height),
            int(width),
        )

    def _sanitize_background_image(self, image: np.ndarray) -> np.ndarray:
        if not self.scenario_path:
            return image
        try:
            data = Path(self.scenario_path).read_bytes()
        except Exception as exc:
            print(f"[WARN] Failed to preprocess background objects from {self.scenario_path}: {exc}")
            return image

        erase_tiles: set[tuple[int, int]] = set()

        if parse_stacks is not None:
            try:
                for stack in parse_stacks(data):
                    erase_tiles.add((int(stack["x"]), int(stack["y"])))
            except Exception as exc:
                print(f"[WARN] Failed to parse stacks from {self.scenario_path}: {exc}")

        if extract_render_objects is not None:
            try:
                render_objects = extract_render_objects(data)
                for class_name, objects in render_objects.items():
                    spec = RENDER_OBJECT_SPECS.get(class_name, {})
                    width, height = spec.get("footprint", (1, 1))
                    for obj in objects:
                        ox = int(obj["x"])
                        oy = int(obj["y"])
                        for tx in range(ox, min(self.grid_size, ox + int(width))):
                            for ty in range(oy, min(self.grid_size, oy + int(height))):
                                erase_tiles.add((tx, ty))
            except Exception as exc:
                print(f"[WARN] Failed to parse map objects from {self.scenario_path}: {exc}")

        erase_tiles = {
            self._display_tile(x, y)
            for x, y in erase_tiles
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size
        }

        if not erase_tiles:
            return image

        cleaned = np.array(image, copy=True)
        source = np.array(image, copy=True)
        cell_h = max(1, cleaned.shape[0] // self.grid_size)
        cell_w = max(1, cleaned.shape[1] // self.grid_size)
        tile_median_cache: dict[tuple[int, int], np.ndarray] = {}

        def tile_bounds(tx: int, ty: int) -> tuple[int, int, int, int]:
            x0 = tx * cell_w
            y0 = ty * cell_h
            x1 = min(cleaned.shape[1], x0 + cell_w)
            y1 = min(cleaned.shape[0], y0 + cell_h)
            return x0, y0, x1, y1

        def tile_median(tx: int, ty: int) -> np.ndarray | None:
            key = (tx, ty)
            cached = tile_median_cache.get(key)
            if cached is not None:
                return cached
            x0, y0, x1, y1 = tile_bounds(tx, ty)
            tile = source[y0:y1, x0:x1]
            if tile.size == 0:
                return None
            median = np.median(tile.reshape(-1, tile.shape[-1]), axis=0)
            tile_median_cache[key] = median
            return median

        for x, y in sorted(erase_tiles):
            if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                continue

            samples = []
            for radius in range(1, 6):
                for nx in range(max(0, x - radius), min(self.grid_size, x + radius + 1)):
                    for ny in range(max(0, y - radius), min(self.grid_size, y + radius + 1)):
                        if max(abs(nx - x), abs(ny - y)) != radius:
                            continue
                        if (nx, ny) in erase_tiles:
                            continue
                        sample = tile_median(nx, ny)
                        if sample is not None:
                            samples.append(sample)
                if samples:
                    break

            fill_color = (
                np.median(np.stack(samples, axis=0), axis=0)
                if samples
                else tile_median(x, y)
            )
            if fill_color is None:
                continue
            x0, y0, x1, y1 = tile_bounds(x, y)
            cleaned[y0:y1, x0:x1] = fill_color

        return cleaned

    def _load_settlement_overlays(self) -> None:
        if not self.scenario_path or parse_settlement_objects is None:
            return
        try:
            data = Path(self.scenario_path).read_bytes()
            settlement_objects = parse_settlement_objects(data)
        except Exception as exc:
            print(f"[WARN] Failed to load settlement overlays from {self.scenario_path}: {exc}")
            return

        overlays = []
        for settlement in settlement_objects.values():
            class_name = str(settlement.get("class") or "")
            spec = RENDER_OBJECT_SPECS.get(class_name, {})
            width, height = spec.get("footprint", (1, 1))
            try:
                x = int(settlement["x"])
                y = int(settlement["y"])
            except Exception:
                continue
            overlays.append(
                {
                    "class": class_name,
                    "x": x,
                    "y": y,
                    "width": int(width),
                    "height": int(height),
                    "entry_x": x,
                    "entry_y": y + max(0, int(height) - 1),
                    "diagonal_x": x + max(0, int(width) - 1),
                    "diagonal_y": y,
                }
            )

        self.capital_overlays = [
            overlay for overlay in overlays if overlay["class"] == ".?AVCCapital@@"
        ]
        self.village_overlays = [
            overlay for overlay in overlays if overlay["class"] == ".?AVCMidVillage@@"
        ]

    def _draw_background_map(self) -> None:
        if self.background_image is None:
            return
        self.ax.imshow(
            self.background_image,
            extent=(-0.5, self.grid_size - 0.5, -0.5, self.grid_size - 0.5),
            origin="lower",
            interpolation="nearest",
            zorder=0,
        )

    def _draw_settlement_overlays(self) -> None:
        for overlay in self.capital_overlays:
            x = int(overlay["x"])
            y = int(overlay["y"])
            width = int(overlay["width"])
            height = int(overlay["height"])
            draw_x, draw_y, draw_w, draw_h = self._display_rect(x, y, width, height)

            rect = patches.Rectangle(
                (draw_x - 0.5, draw_y - 0.5),
                draw_w,
                draw_h,
                facecolor=self.COLOR_CAPITAL_FILL,
                edgecolor=self.COLOR_CAPITAL_EDGE,
                linewidth=2.5,
                zorder=2.2,
            )
            self.ax.add_patch(rect)

            center_x = draw_x + (draw_w - 1) / 2.0
            center_y = draw_y + (draw_h - 1) / 2.0
            self.ax.text(
                center_x,
                center_y,
                "CAP",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
                zorder=2.5,
            )

    def _load_ruin_overlays(self) -> None:
        if not self.scenario_path or extract_render_objects is None:
            return
        try:
            data = Path(self.scenario_path).read_bytes()
            render_objects = extract_render_objects(data)
        except Exception as exc:
            print(f"[WARN] Failed to load ruin overlays from {self.scenario_path}: {exc}")
            return

        overlays = []
        spec = RENDER_OBJECT_SPECS.get(".?AVCMidRuin@@", {})
        width, height = spec.get("footprint", (3, 3))
        for ruin in render_objects.get(".?AVCMidRuin@@", []):
            try:
                x = int(ruin["x"])
                y = int(ruin["y"])
            except Exception:
                continue
            overlays.append(
                {
                    "x": x,
                    "y": y,
                    "width": int(width),
                    "height": int(height),
                }
            )
        self.ruin_overlays = sorted(overlays, key=lambda overlay: (overlay["y"], overlay["x"]))

    def _load_gold_mine_overlays(self) -> None:
        if not self.scenario_path or extract_render_objects is None:
            return
        try:
            data = Path(self.scenario_path).read_bytes()
            render_objects = extract_render_objects(data)
        except Exception as exc:
            print(f"[WARN] Failed to load gold mine overlays from {self.scenario_path}: {exc}")
            return

        overlays = []
        spec = RENDER_OBJECT_SPECS.get("resource:0", {})
        width, height = spec.get("footprint", (1, 1))
        for gold_mine in render_objects.get("resource:0", []):
            try:
                x = int(gold_mine["x"])
                y = int(gold_mine["y"])
            except Exception:
                continue
            overlays.append(
                {
                    "x": x,
                    "y": y,
                    "width": int(width),
                    "height": int(height),
                }
            )
        self.gold_mine_overlays = sorted(overlays, key=lambda overlay: (overlay["y"], overlay["x"]))

    def _draw_gold_mine_overlays(self) -> None:
        for overlay in self.gold_mine_overlays:
            x = int(overlay["x"])
            y = int(overlay["y"])
            draw_x, draw_y = self._display_tile(x, y)
            gold_rect = patches.Rectangle(
                (draw_x - 0.36, draw_y - 0.36),
                0.72,
                0.72,
                facecolor=self.COLOR_GOLD_MINE_FILL,
                edgecolor=self.COLOR_GOLD_MINE_EDGE,
                linewidth=1.8,
                zorder=2.72,
            )
            self.ax.add_patch(gold_rect)
            self.ax.text(
                draw_x,
                draw_y,
                "З",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=self.COLOR_GOLD_MINE_TEXT,
                zorder=2.8,
            )

    def _draw_mana_source_overlays(
        self,
        mana_sources: dict[tuple[int, int], dict[str, object]] | None,
    ) -> None:
        if not mana_sources:
            return
        for tile, meta in sorted(mana_sources.items(), key=lambda item: (item[0][1], item[0][0])):
            try:
                x = int(tile[0])
                y = int(tile[1])
            except Exception:
                continue
            draw_x, draw_y = self._display_tile(x, y)
            fill_color = tuple(meta.get("fill_color", (0.8, 0.8, 0.8, 0.84)))
            edge_color = tuple(meta.get("edge_color", (0.3, 0.3, 0.3, 0.98)))
            text_color = tuple(meta.get("text_color", (0.05, 0.05, 0.05, 1.0)))
            letter = str(meta.get("letter", "?") or "?")[:1]
            mana_rect = patches.Rectangle(
                (draw_x - 0.36, draw_y - 0.36),
                0.72,
                0.72,
                facecolor=fill_color,
                edgecolor=edge_color,
                linewidth=1.8,
                zorder=2.74,
            )
            self.ax.add_patch(mana_rect)
            self.ax.text(
                draw_x,
                draw_y,
                letter,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=text_color,
                zorder=2.82,
            )

    def _draw_ruin_overlays(self) -> None:
        for overlay in self.ruin_overlays:
            x = int(overlay["x"])
            y = int(overlay["y"])
            width = int(overlay["width"])
            height = int(overlay["height"])
            draw_x, draw_y, draw_w, draw_h = self._display_rect(x, y, width, height)
            ruin_rect = patches.Rectangle(
                (draw_x - 0.5, draw_y - 0.5),
                draw_w,
                draw_h,
                facecolor="none",
                edgecolor=self.COLOR_RUIN_EDGE,
                linewidth=2.0,
                linestyle="--",
                zorder=2.66,
            )
            self.ax.add_patch(ruin_rect)

            marker_x = x + max(0, width - 1)
            marker_y = y + max(0, height - 1)
            coord_draw_x, coord_draw_y = self._display_tile(marker_x, marker_y)
            coord_rect = patches.Rectangle(
                (coord_draw_x - 0.42, coord_draw_y - 0.42),
                0.84,
                0.84,
                facecolor=(0.95, 0.12, 0.12, 0.18),
                edgecolor=self.COLOR_RUIN_EDGE,
                linewidth=2.0,
                zorder=2.74,
            )
            self.ax.add_patch(coord_rect)
            self.ax.text(
                coord_draw_x,
                coord_draw_y,
                "Р",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=self.COLOR_RUIN_EDGE,
                zorder=2.82,
            )

    def draw_grid(
        self,
        agent_pos: tuple,
        enemy_positions: dict,
        enemies_alive: dict,
        visited_cells: set,
        title: str = "",
        highlight_battle: int = None,
        castle_heal_tiles: list | tuple | None = None,
        legions_territory_tiles: list | tuple | set | None = None,
        empire_territory_tiles: list | tuple | set | None = None,
        obstacle_tiles: list | tuple | set | None = None,
        merchant_positions: list | tuple | set | dict | None = None,
        merchant_site_anchors: list | tuple | set | dict | None = None,
        spell_shop_positions: list | tuple | set | dict | None = None,
        spell_shop_site_anchors: list | tuple | set | dict | None = None,
        mercenary_positions: list | tuple | set | dict | None = None,
        mercenary_site_anchors: list | tuple | set | dict | None = None,
        trainer_positions: list | tuple | set | dict | None = None,
        trainer_site_anchors: list | tuple | set | dict | None = None,
        chest_positions: list | tuple | set | dict | None = None,
        mana_sources: dict | None = None,
        mana_totals: dict | None = None,
        mana_income_per_turn: dict | None = None,
        turns: int = 0,
        gold: float = 0.0,
        steps: int = 0,
        heal_bottles_left: int = 0,
        max_heal_bottles: int = 0,
        healing_bottles_left: int = 0,
        max_healing_bottles: int = 0,
        revive_bottles_left: int = 0,
        max_revive_bottles: int = 0,
        built_buildings: list = None,
        learned_spells: list | None = None,
        heroitems: list | None = None,
        equipped_hero_items: list | None = None,
        battle_items_equipped_total: int = 0,
        battle_items_used_total: int = 0,
        hero_name: str = "",
        hero_level: int = 0,
        hero_abilities: list | None = None,
        ruins_cleared: int = 0,
        ruins_total: int = 0,
        last_ruin_reward: dict | None = None,
        scripted_bot: dict | None = None,
    ):
        """Отрисовка карты."""
        self.ax.clear()
        self.info_ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect("equal")
        self.ax.set_facecolor((0.95, 0.95, 0.95))
        self.info_ax.axis("off")
        self.ax.invert_yaxis()  # Y растёт вниз

        # Сетка
        self._draw_background_map()

        for i in range(self.grid_size + 1):
            self.ax.axhline(y=i - 0.5, color="gray", linewidth=0.5, alpha=0.35, zorder=1)
            self.ax.axvline(x=i - 0.5, color="gray", linewidth=0.5, alpha=0.35, zorder=1)

        # Посещённые клетки
        for (x, y) in visited_cells:
            draw_x, draw_y = self._display_tile(int(x), int(y))
            rect = patches.Rectangle(
                (draw_x - 0.45, draw_y - 0.45), 0.9, 0.9,
                facecolor=self.COLOR_VISITED,
                edgecolor="none",
            )
            self.ax.add_patch(rect)

        if legions_territory_tiles:
            for tile in legions_territory_tiles:
                try:
                    lx, ly = int(tile[0]), int(tile[1])
                except Exception:
                    continue
                draw_x, draw_y = self._display_tile(lx, ly)
                territory_rect = patches.Rectangle(
                    (draw_x - 0.45, draw_y - 0.45), 0.9, 0.9,
                    facecolor=self.COLOR_LEGIONS_TERRITORY,
                    edgecolor="none",
                    zorder=1.6,
                )
                self.ax.add_patch(territory_rect)

        if empire_territory_tiles:
            for tile in empire_territory_tiles:
                try:
                    ex, ey = int(tile[0]), int(tile[1])
                except Exception:
                    continue
                draw_x, draw_y = self._display_tile(ex, ey)
                territory_rect = patches.Rectangle(
                    (draw_x - 0.45, draw_y - 0.45), 0.9, 0.9,
                    facecolor=self.COLOR_EMPIRE_TERRITORY,
                    edgecolor="none",
                    zorder=1.62,
                )
                self.ax.add_patch(territory_rect)

        if obstacle_tiles:
            for tile in obstacle_tiles:
                try:
                    ox, oy = int(tile[0]), int(tile[1])
                except Exception:
                    continue
                draw_x, draw_y = self._display_tile(ox, oy)
                obstacle_rect = patches.Rectangle(
                    (draw_x - 0.45, draw_y - 0.45), 0.9, 0.9,
                    facecolor=self.COLOR_OBSTACLE,
                    edgecolor=(0.25, 0.25, 0.25),
                    linewidth=1.5,
                )
                self.ax.add_patch(obstacle_rect)

        self._draw_settlement_overlays()

        # Heal tiles on the map — highlight in yellow.
        if castle_heal_tiles is None:
            heal_tiles = [self.CASTLE_POS]
        else:
            heal_tiles = list(castle_heal_tiles)
            if not heal_tiles:
                heal_tiles = [self.CASTLE_POS]
        seen_tiles = set()
        for tile in heal_tiles:
            try:
                cx, cy = int(tile[0]), int(tile[1])
            except Exception:
                continue
            draw_x, draw_y = self._display_tile(cx, cy)
            if (draw_x, draw_y) in seen_tiles:
                continue
            seen_tiles.add((draw_x, draw_y))
            castle_rect = patches.Rectangle(
                (draw_x - 0.45, draw_y - 0.45), 0.9, 0.9,
                facecolor=self.COLOR_CASTLE,
                edgecolor=(0.9, 0.6, 0.0),
                linewidth=2.0,
            )
            self.ax.add_patch(castle_rect)
            self.ax.text(
                draw_x, draw_y, "C",
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color=(0.35, 0.25, 0.0),
            )

        if merchant_site_anchors:
            raw_merchant_anchors = (
                merchant_site_anchors.values()
                if isinstance(merchant_site_anchors, dict)
                else merchant_site_anchors
            )
            for anchor in raw_merchant_anchors:
                try:
                    anchor_x, anchor_y = int(anchor[0]), int(anchor[1])
                except Exception:
                    continue
                draw_y, draw_x, draw_h, draw_w = self._display_rect(anchor_x, anchor_y, 3, 3)
                marker_x = anchor_x + 2
                marker_y = anchor_y + 2
                anchor_draw_x, anchor_draw_y = self._display_tile(marker_x, marker_y)
                merchant_anchor_rect = patches.Rectangle(
                    (anchor_draw_x - 0.45, anchor_draw_y - 0.45),
                    0.9,
                    0.9,
                    facecolor=self.COLOR_MERCHANT_ANCHOR_FILL,
                    edgecolor=self.COLOR_MERCHANT_ANCHOR_EDGE,
                    linewidth=2.2,
                    zorder=2.74,
                )
                self.ax.add_patch(merchant_anchor_rect)
                self.ax.text(
                    anchor_draw_x,
                    anchor_draw_y,
                    "T",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                    zorder=2.82,
                )

        if spell_shop_site_anchors:
            raw_spell_shop_anchors = (
                spell_shop_site_anchors.values()
                if isinstance(spell_shop_site_anchors, dict)
                else spell_shop_site_anchors
            )
            for anchor in raw_spell_shop_anchors:
                try:
                    anchor_x, anchor_y = int(anchor[0]), int(anchor[1])
                except Exception:
                    continue
                marker_x = anchor_x + 2
                marker_y = anchor_y + 2
                anchor_draw_x, anchor_draw_y = self._display_tile(marker_x, marker_y)
                spell_shop_anchor_rect = patches.Rectangle(
                    (anchor_draw_x - 0.45, anchor_draw_y - 0.45),
                    0.9,
                    0.9,
                    facecolor=self.COLOR_SPELL_SHOP_ANCHOR_FILL,
                    edgecolor=self.COLOR_SPELL_SHOP_ANCHOR_EDGE,
                    linewidth=2.2,
                    zorder=2.745,
                )
                self.ax.add_patch(spell_shop_anchor_rect)
                self.ax.text(
                    anchor_draw_x,
                    anchor_draw_y,
                    "S",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                    zorder=2.825,
                )

        if mercenary_site_anchors:
            raw_mercenary_anchors = (
                mercenary_site_anchors.values()
                if isinstance(mercenary_site_anchors, dict)
                else mercenary_site_anchors
            )
            for anchor in raw_mercenary_anchors:
                try:
                    anchor_x, anchor_y = int(anchor[0]), int(anchor[1])
                except Exception:
                    continue
                marker_x = anchor_x + 2
                marker_y = anchor_y + 2
                anchor_draw_x, anchor_draw_y = self._display_tile(marker_x, marker_y)
                mercenary_anchor_rect = patches.Rectangle(
                    (anchor_draw_x - 0.45, anchor_draw_y - 0.45),
                    0.9,
                    0.9,
                    facecolor=self.COLOR_MERCENARY_ANCHOR_FILL,
                    edgecolor=self.COLOR_MERCENARY_ANCHOR_EDGE,
                    linewidth=2.2,
                    zorder=2.75,
                )
                self.ax.add_patch(mercenary_anchor_rect)
                self.ax.text(
                    anchor_draw_x,
                    anchor_draw_y,
                    "H",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                    zorder=2.83,
                )

        if trainer_site_anchors:
            raw_trainer_anchors = (
                trainer_site_anchors.values()
                if isinstance(trainer_site_anchors, dict)
                else trainer_site_anchors
            )
            for anchor in raw_trainer_anchors:
                try:
                    anchor_x, anchor_y = int(anchor[0]), int(anchor[1])
                except Exception:
                    continue
                marker_x = anchor_x + 2
                marker_y = anchor_y + 2
                anchor_draw_x, anchor_draw_y = self._display_tile(marker_x, marker_y)
                trainer_anchor_rect = patches.Rectangle(
                    (anchor_draw_x - 0.45, anchor_draw_y - 0.45),
                    0.9,
                    0.9,
                    facecolor=self.COLOR_TRAINER_ANCHOR_FILL,
                    edgecolor=self.COLOR_TRAINER_ANCHOR_EDGE,
                    linewidth=2.2,
                    zorder=2.76,
                )
                self.ax.add_patch(trainer_anchor_rect)
                self.ax.text(
                    anchor_draw_x,
                    anchor_draw_y,
                    "R",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                    zorder=2.84,
                )

        # Путь агента
        if merchant_positions:
            raw_merchant_tiles = (
                merchant_positions.keys()
                if isinstance(merchant_positions, dict)
                else merchant_positions
            )
            for tile in raw_merchant_tiles:
                try:
                    mx, my = int(tile[0]), int(tile[1])
                except Exception:
                    continue
                draw_x, draw_y = self._display_tile(mx, my)
                merchant_rect = patches.Rectangle(
                    (draw_x - 0.40, draw_y - 0.40), 0.80, 0.80,
                    facecolor=self.COLOR_MERCHANT_FILL,
                    edgecolor=self.COLOR_MERCHANT_EDGE,
                    linewidth=1.2,
                    linestyle="--",
                    zorder=2.65,
                )
                self.ax.add_patch(merchant_rect)

        if spell_shop_positions:
            raw_spell_shop_tiles = (
                spell_shop_positions.keys()
                if isinstance(spell_shop_positions, dict)
                else spell_shop_positions
            )
            for tile in raw_spell_shop_tiles:
                try:
                    sx, sy = int(tile[0]), int(tile[1])
                except Exception:
                    continue
                draw_x, draw_y = self._display_tile(sx, sy)
                spell_shop_rect = patches.Rectangle(
                    (draw_x - 0.40, draw_y - 0.40), 0.80, 0.80,
                    facecolor=self.COLOR_SPELL_SHOP_FILL,
                    edgecolor=self.COLOR_SPELL_SHOP_EDGE,
                    linewidth=1.2,
                    linestyle="--",
                    zorder=2.655,
                )
                self.ax.add_patch(spell_shop_rect)

        if mercenary_positions:
            raw_mercenary_tiles = (
                mercenary_positions.keys()
                if isinstance(mercenary_positions, dict)
                else mercenary_positions
            )
            for tile in raw_mercenary_tiles:
                try:
                    mx, my = int(tile[0]), int(tile[1])
                except Exception:
                    continue
                draw_x, draw_y = self._display_tile(mx, my)
                mercenary_rect = patches.Rectangle(
                    (draw_x - 0.40, draw_y - 0.40), 0.80, 0.80,
                    facecolor=self.COLOR_MERCENARY_FILL,
                    edgecolor=self.COLOR_MERCENARY_EDGE,
                    linewidth=1.2,
                    linestyle="--",
                    zorder=2.66,
                )
                self.ax.add_patch(mercenary_rect)

        if trainer_positions:
            raw_trainer_tiles = (
                trainer_positions.keys()
                if isinstance(trainer_positions, dict)
                else trainer_positions
            )
            for tile in raw_trainer_tiles:
                try:
                    tx, ty = int(tile[0]), int(tile[1])
                except Exception:
                    continue
                draw_x, draw_y = self._display_tile(tx, ty)
                trainer_rect = patches.Rectangle(
                    (draw_x - 0.40, draw_y - 0.40), 0.80, 0.80,
                    facecolor=self.COLOR_TRAINER_FILL,
                    edgecolor=self.COLOR_TRAINER_EDGE,
                    linewidth=1.2,
                    linestyle="--",
                    zorder=2.67,
                )
                self.ax.add_patch(trainer_rect)

        if chest_positions:
            raw_chest_tiles = (
                chest_positions.keys()
                if isinstance(chest_positions, dict)
                else chest_positions
            )
            for tile in raw_chest_tiles:
                try:
                    cx, cy = int(tile[0]), int(tile[1])
                except Exception:
                    continue
                draw_x, draw_y = self._display_tile(cx, cy)
                chest_rect = patches.Rectangle(
                    (draw_x - 0.35, draw_y - 0.30), 0.70, 0.60,
                    facecolor=self.COLOR_CHEST_FILL,
                    edgecolor=self.COLOR_CHEST_EDGE,
                    linewidth=1.8,
                    zorder=2.7,
                )
                self.ax.add_patch(chest_rect)
                self.ax.text(
                    draw_x, draw_y, "T",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color=(1.0, 0.96, 0.72),
                    zorder=2.8,
                )

        if len(self.path_history) > 1:
            for i in range(len(self.path_history) - 1):
                x1, y1 = self.path_history[i]
                x2, y2 = self.path_history[i + 1]
                draw_x1, draw_y1 = self._display_tile(int(x1), int(y1))
                draw_x2, draw_y2 = self._display_tile(int(x2), int(y2))
                self.ax.plot(
                    [draw_x1, draw_x2], [draw_y1, draw_y2],
                    color=self.COLOR_PATH,
                    linewidth=2,
                    alpha=0.7,
                )

        self._draw_gold_mine_overlays()
        self._draw_mana_source_overlays(mana_sources)
        self._draw_ruin_overlays()

        bot_data = dict(scripted_bot or {})
        bot_enabled = bool(bot_data.get("enabled", False))
        bot_pos_raw = bot_data.get("position")
        bot_pos = None
        if bot_enabled and isinstance(bot_pos_raw, (list, tuple)) and len(bot_pos_raw) >= 2:
            try:
                bot_pos = (int(bot_pos_raw[0]), int(bot_pos_raw[1]))
            except Exception:
                bot_pos = None

        # Враги
        for enemy_id, (ex, ey) in enemy_positions.items():
            draw_x, draw_y = self._display_tile(int(ex), int(ey))
            is_alive = enemies_alive.get(enemy_id, False)
            is_final_objective = CampaignEnv.is_final_objective_enemy_id(enemy_id)
            if is_final_objective:
                dragon_tile = patches.Rectangle(
                    (draw_x - 0.48, draw_y - 0.48),
                    0.96,
                    0.96,
                    facecolor=self.COLOR_DRAGON_TILE,
                    edgecolor=(0.05, 0.45, 0.12),
                    linewidth=1.8,
                    zorder=2.6,
                )
                self.ax.add_patch(dragon_tile)
            color = (
                self.COLOR_DRAGON_ALIVE if (is_alive and is_final_objective)
                else self.COLOR_DRAGON_DEAD if is_final_objective
                else self.COLOR_ENEMY_ALIVE if is_alive
                else self.COLOR_ENEMY_DEAD
            )

            # Подсветка текущего боя
            if highlight_battle == enemy_id:
                circle = plt.Circle(
                    (draw_x, draw_y), 0.55,
                    facecolor="none",
                    edgecolor=(1.0, 0.8, 0.0),
                    linewidth=4,
                )
                self.ax.add_patch(circle)

            circle = plt.Circle((draw_x, draw_y), 0.35, facecolor=color, edgecolor="black", linewidth=1.5)
            self.ax.add_patch(circle)

            # Номер врага
            status = "OK" if not is_alive else ""
            self.ax.text(
                draw_x, draw_y, f"E{enemy_id}{status}",
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if is_alive else "gray",
            )

        if bot_pos is not None:
            draw_bot_x, draw_bot_y = self._display_tile(int(bot_pos[0]), int(bot_pos[1]))
            bot_circle = plt.Circle(
                (draw_bot_x, draw_bot_y),
                0.34,
                facecolor=self.COLOR_SCRIPTED_BOT,
                edgecolor=(0.22, 0.03, 0.35),
                linewidth=2,
                zorder=3.1,
            )
            self.ax.add_patch(bot_circle)
            self.ax.text(
                draw_bot_x,
                draw_bot_y,
                str(bot_data.get("symbol", "B") or "B")[:1],
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white",
                zorder=3.2,
            )

        # Агент
        draw_agent_x, draw_agent_y = self._display_tile(int(agent_pos[0]), int(agent_pos[1]))
        agent_circle = plt.Circle(
            (draw_agent_x, draw_agent_y), 0.3,
            facecolor=self.COLOR_AGENT,
            edgecolor="darkblue",
            linewidth=2,
        )
        self.ax.add_patch(agent_circle)
        self.ax.text(
            draw_agent_x, draw_agent_y, "A",
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color="white",
        )

        # Заголовок
        self.ax.set_title(title, fontsize=14, fontweight="bold")

        # Ход, золото, шаги и постройки справа от грида
        bot_panel_early = dict(scripted_bot or {})
        try:
            bot_victims_count = int(bot_panel_early.get("enemies_defeated", 0) or 0)
        except (TypeError, ValueError):
            bot_victims_count = 0

        info_lines = [
            f"Ход: {turns}",
            f"Золото: {gold:g}",
            f"Шаги: {steps}",
            f"Леч. бутылки: {heal_bottles_left}/{max_heal_bottles}",
            f"Исц. бутылки: {healing_bottles_left}/{max_healing_bottles}",
            f"Воскр. бутылки: {revive_bottles_left}/{max_revive_bottles}",
        ]
        built_buildings = built_buildings or []
        if built_buildings:
            info_lines.append("Постройки:")
            info_lines.extend(f"- {name}" for name in built_buildings)
        else:
            info_lines.append("Постройки: нет")
        learned_spells = learned_spells or []
        if learned_spells:
            info_lines.append("Заклинания:")
            info_lines.extend(f"- {description}" for description in learned_spells)
        else:
            info_lines.append("Заклинания: нет")
        merchant_count = len(merchant_positions) if merchant_positions else 0
        info_lines.append(f"Merchant tiles: {merchant_count}")
        spell_shop_count = len(spell_shop_positions) if spell_shop_positions else 0
        info_lines.append(f"Spell shop tiles: {spell_shop_count}")
        mercenary_count = len(mercenary_positions) if mercenary_positions else 0
        info_lines.append(f"Mercenary tiles: {mercenary_count}")
        trainer_count = len(trainer_positions) if trainer_positions else 0
        info_lines.append(f"Trainer tiles: {trainer_count}")
        chest_count = len(chest_positions) if chest_positions else 0
        legions_territory_count = len(legions_territory_tiles) if legions_territory_tiles else 0
        empire_territory_count = len(empire_territory_tiles) if empire_territory_tiles else 0
        info_lines.append(f"Legions territory: {legions_territory_count}")
        info_lines.append(f"Empire territory: {empire_territory_count}")
        info_lines.append(f"Сундуки: {chest_count}")
        mana_totals = dict(mana_totals or {})
        mana_income_per_turn = dict(mana_income_per_turn or {})
        mana_display_order = (
            ("infernal", "Мана преисподней"),
            ("life", "Мана жизни"),
            ("death", "Мана смерти"),
            ("runes", "Мана рун"),
            ("elves", "Мана эльфов"),
        )
        info_lines.append("Мана:")
        for mana_kind, mana_label in mana_display_order:
            try:
                mana_total = float(mana_totals.get(mana_kind, 0.0) or 0.0)
            except (TypeError, ValueError):
                mana_total = 0.0
            try:
                mana_delta = float(mana_income_per_turn.get(mana_kind, 0.0) or 0.0)
            except (TypeError, ValueError):
                mana_delta = 0.0
            info_lines.append(f"- {mana_label}: {mana_total:g} (+{mana_delta:g}/ход)")
        try:
            ruins_cleared = max(0, int(ruins_cleared or 0))
        except (TypeError, ValueError):
            ruins_cleared = 0
        try:
            ruins_total = max(0, int(ruins_total or 0))
        except (TypeError, ValueError):
            ruins_total = 0
        if ruins_total > 0:
            info_lines.append(f"Руины: {ruins_cleared}/{ruins_total}")
            ruin_reward_data = dict(last_ruin_reward or {})
            if ruin_reward_data:
                ruin_title = str(ruin_reward_data.get("title", "") or "").strip()
                if not ruin_title:
                    ruin_pos = ruin_reward_data.get("ruin_position")
                    if isinstance(ruin_pos, (list, tuple)) and len(ruin_pos) >= 2:
                        ruin_title = f"Руины ({int(ruin_pos[0])}, {int(ruin_pos[1])})"
                ruin_item = str(ruin_reward_data.get("item", "") or "").strip()
                try:
                    ruin_gold = max(0.0, float(ruin_reward_data.get("gold", 0.0) or 0.0))
                except (TypeError, ValueError):
                    ruin_gold = 0.0
                if ruin_title:
                    info_lines.append(f"Последняя руина: {ruin_title}")
                reward_parts = []
                if ruin_gold > 0.0:
                    reward_parts.append(f"+{ruin_gold:g} gold")
                if ruin_item:
                    reward_parts.append(ruin_item)
                if reward_parts:
                    info_lines.append(f"Добыча руин: {', '.join(reward_parts)}")
            else:
                info_lines.append("Последняя руина: нет")
        def _hero_item_label(entry):
            if isinstance(entry, dict):
                return str(entry.get("name", "") or "")
            return str(entry or "")

        heroitems = list(heroitems or [])
        if heroitems:
            info_lines.append("Hero items:")
            info_lines.extend(f"- {_hero_item_label(item_name)}" for item_name in heroitems)
        else:
            info_lines.append("Hero items: none")
        equipped_hero_items = list(equipped_hero_items or [])
        if equipped_hero_items:
            info_lines.append("Equipped battle items:")
            for slot_index, item_name in enumerate(equipped_hero_items, start=1):
                label = _hero_item_label(item_name) if item_name else "none"
                info_lines.append(f"- slot {slot_index}: {label}")
        else:
            info_lines.append("Equipped battle items: none")
        info_lines.append(
            f"Battle items stats: equipped={int(battle_items_equipped_total)} "
            f"used={int(battle_items_used_total)}"
        )
        hero_name = str(hero_name or "").strip()
        if bot_enabled:
            info_lines.append(
                f"Бот столицы: {bot_data.get('state', 'unknown')} "
                f"@ {tuple(bot_pos) if bot_pos is not None else '?'}"
            )
        hero_abilities = list(hero_abilities or [])
        if hero_name:
            info_lines.append(f"Герой: {hero_name} (ур. {hero_level})")
            info_lines.append("Способности героя:")
            if hero_abilities:
                info_lines.extend(f"- {ability}" for ability in hero_abilities)
            else:
                info_lines.append("- нет")
        info_text = self._format_info_text(info_lines)
        bot_highlight = f"Врагов победил бот (B): {bot_victims_count}"
        self._draw_info_panel(title=title, info_text=info_text, highlight_line=bot_highlight)

        # Легенда
        legend_text = f"Visited: {len(visited_cells)}/{self.grid_size**2}"
        alive_count = sum(1 for v in enemies_alive.values() if v)
        total_enemies = len(enemies_alive)
        defeated_count = total_enemies - alive_count
        legend_text += f" | Enemies: {defeated_count}/{total_enemies} defeated"
        legend_text += f" | Legions: {legions_territory_count}"
        legend_text += f" | Empire: {empire_territory_count}"
        self.ax.set_xlabel(legend_text, fontsize=11)

        # Убираем оси
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.tick_params(labelsize=8)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def add_to_path(self, pos: tuple):
        """Добавляет позицию в историю пути."""
        if not self.path_history or self.path_history[-1] != pos:
            self.path_history.append(pos)

    def clear_path(self):
        """Очищает историю пути."""
        self.path_history = []

    def show_battle_start(self, enemy_id: int):
        """Показывает начало боя."""
        self.ax.text(
            self.grid_size / 2, self.grid_size / 2,
            f"⚔️ BATTLE vs Enemy {enemy_id}! ⚔️",
            ha="center", va="center",
            fontsize=16, fontweight="bold",
            color="red",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show_battle_result(self, won: bool, enemy_id: int):
        """Показывает результат боя."""
        if won:
            text = f"VICTORY vs Enemy {enemy_id}!"
            color = "green"
        else:
            text = f"❌ DEFEAT vs Enemy {enemy_id}!"
            color = "red"

        self.ax.text(
            self.grid_size / 2, self.grid_size / 2,
            text,
            ha="center", va="center",
            fontsize=16, fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)


def run_campaign_visualization(
    model_path: str,
    delay: float = 0.3,
    battle_speed: float = 1.0,
    deterministic: bool = False,
    map_png_path: str | None = None,
    scenario_path: str | None = None,
    scripted_capital_bot_enabled: bool = True,
    campaign_objective: str = "cities",
    vecnormalize_path: str | None = None,
):
    """Запускает визуализацию кампании."""

    # Загрузка модели
    if model_path and os.path.exists(model_path):
        print(f"Загрузка модели из {model_path}")
        model = MaskablePPO.load(model_path)
    else:
        print(f"Модель не найдена: {model_path}")
        print("Используется случайный агент.")

        class RandomModel:
            def predict(self, obs, deterministic=True, action_masks=None):
                valid = [i for i, m in enumerate(action_masks) if m]
                import random
                return np.array(random.choice(valid)), None

        model = RandomModel()

    # Создание среды
    def make_campaign_env(*, log_enabled: bool) -> CampaignEnv:
        return CampaignEnv(
            grid_size=DEFAULT_GRID_SIZE,
            **REWARD_CONFIG,
            persist_blue_hp=True,
            log_enabled=log_enabled,
            detailed_step_info=log_enabled,
            freeze_dynamic_action_layout=True,
            scripted_capital_bot_enabled=scripted_capital_bot_enabled,
            empire_territory_enabled=True,
            campaign_objective=campaign_objective,
            max_grid_steps=1800,
            Realcapital=2,
        )

    env_base = make_campaign_env(log_enabled=True)
    env = ActionMasker(env_base, mask_fn)
    obs_normalizer = None
    if vecnormalize_path:
        if os.path.exists(vecnormalize_path):
            norm_env = DummyVecEnv(
                [
                    lambda: Monitor(
                        ActionMasker(make_campaign_env(log_enabled=False), mask_fn)
                    )
                ]
            )
            obs_normalizer = VecNormalize.load(vecnormalize_path, norm_env)
            obs_normalizer.training = False
            obs_normalizer.norm_reward = False
            print(f"VecNormalize stats: {vecnormalize_path}")
        else:
            print(f"VecNormalize stats not found: {vecnormalize_path}")

    # Визуализаторы
    grid_viz = CampaignVisualizer(
        grid_size=env_base.grid_size,
        background_map_path=map_png_path or _existing_path_or_none(DEFAULT_MAP_PNG_PATH),
        scenario_path=scenario_path or _existing_path_or_none(DEFAULT_SCENARIO_PATH),
    )
    battle_viz = None  # Создаётся при входе в бой

    print("\n" + "=" * 60)
    print("CAMPAIGN VISUALIZATION")
    print("=" * 60)

    obs, info = env.reset()
    grid_viz.clear_path()
    grid_viz.add_to_path(env_base.grid_env.agent_pos)

    done = False
    total_reward = 0.0
    step_count = 0
    battles_won = 0
    total_enemies = len(env_base.grid_env.enemies_alive)
    current_mode = "grid"
    battle_logs_buffer = []

    def snapshot_blue_names_by_pos() -> dict:
        """Capture BLUE unit names by position for upgrade diffing."""
        if hasattr(env_base, "_get_blue_state"):
            blue_units = env_base._get_blue_state()
        else:
            blue_units = getattr(env_base, "blue_team_state", []) or []

        names_by_pos = {}
        for unit in blue_units:
            pos = unit.get("position")
            if pos is None:
                continue
            try:
                pos_i = int(pos)
            except Exception:
                continue
            names_by_pos[pos_i] = str(unit.get("name", "") or "").strip()
        return names_by_pos

    prev_blue_names = snapshot_blue_names_by_pos()

    def get_bottle_counters():
        if hasattr(env_base, "get_bottle_inventory_counters"):
            counters = env_base.get_bottle_inventory_counters()
            return (
                int(counters.get("heal_left", 0) or 0),
                int(counters.get("max_heal", 0) or 0),
                int(counters.get("healing_left", 0) or 0),
                int(counters.get("max_healing", 0) or 0),
                int(counters.get("revive_left", 0) or 0),
                int(counters.get("max_revive", 0) or 0),
            )

        max_heal = int(getattr(env_base, "MAX_HEAL_BOTTLES", 0) or 0)
        max_healing = int(getattr(env_base, "MAX_HEALING_BOTTLES", 0) or 0)
        max_revive = int(getattr(env_base, "MAX_REVIVE_BOTTLES", 0) or 0)
        heal_used = int(getattr(env_base, "heal_bottles_used", 0) or 0)
        healing_used = int(getattr(env_base, "healing_bottles_used", 0) or 0)
        revive_used = int(getattr(env_base, "revive_bottles_used", 0) or 0)
        heal_left = max(0, max_heal - heal_used)
        healing_left = max(0, max_healing - healing_used)
        revive_left = max(0, max_revive - revive_used)
        return heal_left, max_heal, healing_left, max_healing, revive_left, max_revive

    def get_travel_hero_panel():
        if not hasattr(env_base, "get_travel_hero_visual_info"):
            return "", 0, []
        info = env_base.get_travel_hero_visual_info()
        if not info:
            return "", 0, []
        return (
            str(info.get("name", "") or "").strip(),
            int(info.get("level", 0) or 0),
            list(info.get("abilities", []) or []),
        )

    while not done:
        step_count += 1

        # Получаем маску и действие
        mask = env_base.compute_action_mask()

        # Совместимость: если размер наблюдения среды отличается от ожидаемого моделью
        obs_for_model = obs
        if obs_normalizer is not None:
            obs_for_model = obs_normalizer.normalize_obs(
                np.asarray(obs, dtype=np.float32).reshape(1, -1)
            )
            mask_for_model = np.asarray(mask, dtype=bool).reshape(1, -1)
        else:
            mask_for_model = mask
        try:
            target_shape = getattr(model, "observation_space", None)
            exp_len = None
            if target_shape is not None and getattr(target_shape, "shape", None):
                shape = target_shape.shape
                if len(shape) == 1:
                    exp_len = shape[0]
            if exp_len is not None:
                if obs.ndim == 1:
                    if obs.shape[0] > exp_len:
                        obs_for_model = obs[:exp_len]
                    elif obs.shape[0] < exp_len:
                        pad = np.zeros(exp_len - obs.shape[0], dtype=obs.dtype)
                        obs_for_model = np.concatenate([obs, pad], axis=0)
                elif obs.ndim == 2 and obs.shape[1] != exp_len:
                    if obs.shape[1] > exp_len:
                        obs_for_model = obs[:, :exp_len]
                    else:
                        pad = np.zeros((obs.shape[0], exp_len - obs.shape[1]), dtype=obs.dtype)
                        obs_for_model = np.concatenate([obs, pad], axis=1)
        except Exception:
            obs_for_model = obs

        action, _ = model.predict(
            obs_for_model,
            deterministic=deterministic,
            action_masks=mask_for_model,
        )
        action_idx = int(np.asarray(action).reshape(-1)[0])

        # Логируем действие агента в режиме боя
        if current_mode == "battle" and env_base.battle_env:
            if action_idx == DEFEND_ACTION_INDEX:
                chosen_line = f"[STEP {step_count}] Агент выбирает action={action_idx} > защита (+{DEFEND_ARMOR_BONUS} брони)"
            elif action_idx == WAIT_ACTION_INDEX:
                chosen_line = f"[STEP {step_count}] Агент выбирает action={action_idx} > ожидание (сброс инициативы)"
            elif action_idx == RUN_AWAY_ACTION_INDEX:
                chosen_line = f"[STEP {step_count}] Agent chooses action={action_idx} > run_away (running_away=1)"
            elif action_idx < len(TARGET_POSITIONS):
                target_pos = TARGET_POSITIONS[action_idx]
                chosen_line = f"[STEP {step_count}] Агент выбирает action={action_idx} > атака RED pos{target_pos}"
            elif FIRST_HERO_ITEM_ACTION_START <= action_idx < (
                FIRST_HERO_ITEM_ACTION_START + len(HERO_ITEM_TARGET_SLOTS)
            ):
                slot_no = HERO_ITEM_TARGET_SLOTS[action_idx - FIRST_HERO_ITEM_ACTION_START]
                item_name = ""
                if getattr(env_base.battle_env, "equipped_hero_items", None):
                    item_name = str(env_base.battle_env.equipped_hero_items[0] or "")
                chosen_line = (
                    f"[STEP {step_count}] Агент выбирает action={action_idx} > "
                    f"предмет 1 на свой слот {slot_no}"
                )
                chosen_line = (
                    f"[STEP {step_count}] Agent chooses action={action_idx} > "
                    f"item 1 ({item_name or 'empty'}) on own slot {slot_no}"
                )
            elif SECOND_HERO_ITEM_ACTION_START <= action_idx < (
                SECOND_HERO_ITEM_ACTION_START + len(HERO_ITEM_TARGET_SLOTS)
            ):
                slot_no = HERO_ITEM_TARGET_SLOTS[action_idx - SECOND_HERO_ITEM_ACTION_START]
                item_name = ""
                if getattr(env_base.battle_env, "equipped_hero_items", None):
                    item_name = str(env_base.battle_env.equipped_hero_items[1] or "")
                chosen_line = (
                    f"[STEP {step_count}] Агент выбирает action={action_idx} > "
                    f"предмет 2 на свой слот {slot_no}"
                )
                chosen_line = (
                    f"[STEP {step_count}] Agent chooses action={action_idx} > "
                    f"item 2 ({item_name or 'empty'}) on own slot {slot_no}"
                )
            elif FIRST_HERO_ITEM_ENEMY_ACTION_START <= action_idx < (
                FIRST_HERO_ITEM_ENEMY_ACTION_START + len(HERO_ITEM_TARGET_SLOTS)
            ):
                slot_no = HERO_ITEM_TARGET_SLOTS[
                    action_idx - FIRST_HERO_ITEM_ENEMY_ACTION_START
                ]
                item_name = ""
                if getattr(env_base.battle_env, "equipped_hero_items", None):
                    item_name = str(env_base.battle_env.equipped_hero_items[0] or "")
                chosen_line = (
                    f"[STEP {step_count}] Agent chooses action={action_idx} > "
                    f"item 1 ({item_name or 'empty'}) on enemy slot {slot_no}"
                )
            elif SECOND_HERO_ITEM_ENEMY_ACTION_START <= action_idx < (
                SECOND_HERO_ITEM_ENEMY_ACTION_START + len(HERO_ITEM_TARGET_SLOTS)
            ):
                slot_no = HERO_ITEM_TARGET_SLOTS[
                    action_idx - SECOND_HERO_ITEM_ENEMY_ACTION_START
                ]
                item_name = ""
                if getattr(env_base.battle_env, "equipped_hero_items", None):
                    item_name = str(env_base.battle_env.equipped_hero_items[1] or "")
                chosen_line = (
                    f"[STEP {step_count}] Agent chooses action={action_idx} > "
                    f"item 2 ({item_name or 'empty'}) on enemy slot {slot_no}"
                )
            else:
                chosen_line = f"[STEP {step_count}] Агент выбирает action={action_idx}"
            print("\n" + chosen_line)
            battle_logs_buffer.append(chosen_line)

        # Выполняем шаг
        obs, reward, terminated, truncated, info = env.step(action_idx)
        total_reward += reward
        done = terminated or truncated

        # Определяем текущий режим
        new_mode = info.get("mode", "grid")

        # Визуализация в зависимости от режима
        if new_mode == "grid":
            agent_pos = env_base.grid_env.agent_pos
            grid_viz.add_to_path(agent_pos)

            # Заголовок
            action_name = _format_grid_action_label(action, info)
            title = f"Step {step_count} | Action: {action_name} | Reward: {total_reward:.2f}"
            (
                heal_left,
                max_heal,
                healing_left,
                max_healing,
                revive_left,
                max_revive,
            ) = get_bottle_counters()
            hero_name, hero_level, hero_abilities = get_travel_hero_panel()

            grid_viz.draw_grid(
                agent_pos=agent_pos,
                enemy_positions=env_base.grid_env.enemy_positions,
                enemies_alive=env_base.grid_env.enemies_alive,
                visited_cells=env_base.grid_env.visited_cells,
                title=title,
                castle_heal_tiles=getattr(env_base, "castle_heal_tiles", None),
                legions_territory_tiles=getattr(env_base, "legions_territory_tiles", None),
                empire_territory_tiles=getattr(env_base, "empire_territory_tiles", None),
                obstacle_tiles=getattr(env_base.grid_env, "obstacle_positions", None),
                merchant_positions=getattr(env_base.grid_env, "merchant_positions", None),
                merchant_site_anchors=getattr(env_base, "merchant_site_anchors", None),
                spell_shop_positions=getattr(env_base.grid_env, "spell_shop_positions", None),
                spell_shop_site_anchors=getattr(env_base, "spell_shop_site_anchors", None),
                mercenary_positions=getattr(env_base.grid_env, "mercenary_positions", None),
                mercenary_site_anchors=getattr(env_base, "mercenary_site_anchors", None),
                trainer_positions=getattr(env_base.grid_env, "trainer_positions", None),
                trainer_site_anchors=getattr(env_base, "trainer_site_anchors", None),
                mana_sources=getattr(env_base.grid_env, "mana_sources", None),
                mana_totals=env_base._current_mana_totals(),
                mana_income_per_turn=getattr(env_base, "mana_income_per_turn", None),
                turns=env_base.turns,
                gold=env_base.gold,
                steps=env_base.moves,
                heal_bottles_left=heal_left,
                max_heal_bottles=max_heal,
                healing_bottles_left=healing_left,
                max_healing_bottles=max_healing,
                revive_bottles_left=revive_left,
                max_revive_bottles=max_revive,
                built_buildings=env_base.get_built_building_names(),
                learned_spells=env_base.get_learned_spell_descriptions(),
                chest_positions=getattr(env_base, "chests", None),
                heroitems=getattr(env_base, "heroitems", None),
                equipped_hero_items=getattr(env_base, "equipped_hero_items", None),
                battle_items_equipped_total=getattr(env_base, "battle_items_equipped_total", 0),
                battle_items_used_total=getattr(env_base, "battle_items_used_total", 0),
                hero_name=hero_name,
                hero_level=hero_level,
                hero_abilities=hero_abilities,
                ruins_cleared=len(getattr(env_base, "cleared_ruin_enemy_ids", ())),
                ruins_total=len(getattr(env_base, "RUIN_REWARD_BY_ENEMY_ID", {})),
                last_ruin_reward=getattr(env_base, "last_ruin_reward", None),
                scripted_bot=env_base._scripted_capital_bot_info()["scripted_capital_bot"],
            )

            # Проверяем, был ли только что бой
            if current_mode == "battle" and info.get("battle_result"):
                won = info.get("battle_result") == "victory"
                enemy_id = info.get("last_enemy_id") or env_base.current_enemy_id
                grid_viz.show_battle_result(won, enemy_id)
                if won:
                    battles_won += 1

                # Закрываем визуализатор боя
                if battle_viz:
                    battle_viz.close()
                    battle_viz = None
                battle_logs_buffer = []

                time.sleep(delay * 3)

            time.sleep(delay)

        elif new_mode == "battle":
            if current_mode == "grid":
                # Начало боя — показываем на карте
                enemy_id = info.get("enemy_id")
                print(f"\n⚔️ Начало боя с врагом {enemy_id}!")
                print(f"   {ENEMY_DESCRIPTIONS.get(enemy_id, 'Unknown')}")
                (
                    heal_left,
                    max_heal,
                    healing_left,
                    max_healing,
                    revive_left,
                    max_revive,
                ) = get_bottle_counters()
                hero_name, hero_level, hero_abilities = get_travel_hero_panel()

                grid_viz.draw_grid(
                    agent_pos=env_base.grid_env.agent_pos,
                    enemy_positions=env_base.grid_env.enemy_positions,
                    enemies_alive=env_base.grid_env.enemies_alive,
                    visited_cells=env_base.grid_env.visited_cells,
                    title=f"BATTLE vs Enemy {enemy_id}!",
                    highlight_battle=enemy_id,
                    castle_heal_tiles=getattr(env_base, "castle_heal_tiles", None),
                    legions_territory_tiles=getattr(env_base, "legions_territory_tiles", None),
                    empire_territory_tiles=getattr(env_base, "empire_territory_tiles", None),
                    obstacle_tiles=getattr(env_base.grid_env, "obstacle_positions", None),
                    merchant_positions=getattr(env_base.grid_env, "merchant_positions", None),
                    merchant_site_anchors=getattr(env_base, "merchant_site_anchors", None),
                    spell_shop_positions=getattr(env_base.grid_env, "spell_shop_positions", None),
                    spell_shop_site_anchors=getattr(env_base, "spell_shop_site_anchors", None),
                    mercenary_positions=getattr(env_base.grid_env, "mercenary_positions", None),
                    mercenary_site_anchors=getattr(env_base, "mercenary_site_anchors", None),
                    trainer_positions=getattr(env_base.grid_env, "trainer_positions", None),
                    trainer_site_anchors=getattr(env_base, "trainer_site_anchors", None),
                    mana_sources=getattr(env_base.grid_env, "mana_sources", None),
                    mana_totals=env_base._current_mana_totals(),
                    mana_income_per_turn=getattr(env_base, "mana_income_per_turn", None),
                    turns=env_base.turns,
                    gold=env_base.gold,
                    steps=env_base.moves,
                    heal_bottles_left=heal_left,
                    max_heal_bottles=max_heal,
                    healing_bottles_left=healing_left,
                    max_healing_bottles=max_healing,
                    revive_bottles_left=revive_left,
                    max_revive_bottles=max_revive,
                    built_buildings=env_base.get_built_building_names(),
                    learned_spells=env_base.get_learned_spell_descriptions(),
                    chest_positions=getattr(env_base, "chests", None),
                    heroitems=getattr(env_base, "heroitems", None),
                    equipped_hero_items=getattr(env_base, "equipped_hero_items", None),
                    battle_items_equipped_total=getattr(env_base, "battle_items_equipped_total", 0),
                    battle_items_used_total=getattr(env_base, "battle_items_used_total", 0),
                    hero_name=hero_name,
                    hero_level=hero_level,
                    hero_abilities=hero_abilities,
                    ruins_cleared=len(getattr(env_base, "cleared_ruin_enemy_ids", ())),
                    ruins_total=len(getattr(env_base, "RUIN_REWARD_BY_ENEMY_ID", {})),
                    last_ruin_reward=getattr(env_base, "last_ruin_reward", None),
                    scripted_bot=env_base._scripted_capital_bot_info()["scripted_capital_bot"],
                )
                grid_viz.show_battle_start(enemy_id)
                time.sleep(delay * 2)

                # Создаём визуализатор боя
                battle_viz = BattleVisualizer(speed_mult=battle_speed)
                if env_base.battle_env:
                    battle_viz.init_state_from_units(env_base.battle_env.combined)
                    battle_viz.draw_board(headline=f"Бой с врагом {enemy_id} начинается!", active_pos=None)
                    battle_viz.wait(1.0)

                # Получаем начальные логи боя
                if env_base.battle_env:
                    initial_logs = env_base.battle_env.pop_pretty_events()
                    for log in initial_logs:
                        print(log)
                        battle_logs_buffer.append(log)
                        if battle_viz:
                            battle_viz.process_log_line(log)

            # Получаем новые логи боя
            if env_base.battle_env:
                new_logs = env_base.battle_env.pop_pretty_events()
                for log in new_logs:
                    print(log)
                    battle_logs_buffer.append(log)
                    if battle_viz:
                        battle_viz.process_log_line(log)

            # Синхронизируем HP всех юнитов (включая синих) из battle_env
            if battle_viz and env_base.battle_env:
                battle_viz.sync_hp_from_units(env_base.battle_env.combined)
                battle_viz.draw_board(active_pos=battle_viz.active_pos)

        current_mode = new_mode

        # Логи кампании
        if hasattr(env_base, 'pop_campaign_logs'):
            for log in env_base.pop_campaign_logs():
                print(f"[CAMPAIGN] {log}")

        # Проверяем завершение
        if info.get("campaign_result"):
            result = info["campaign_result"]
            print(f"\n{'=' * 60}")
            if result == "victory":
                print("🎉 КАМПАНИЯ ЗАВЕРШЕНА ПОБЕДОЙ! 🎉")
            elif result == "defeat":
                print("💀 КАМПАНИЯ ПРОВАЛЕНА 💀")
            else:
                print(f"Кампания завершена: {result}")
            print(f"Шагов: {step_count}")
            print(f"Награда: {total_reward:.2f}")
            print(f"Побед в боях: {battles_won}/{total_enemies}")
            print(f"{'=' * 60}")

        # Выводим опыт живых юнитов героя после каждой битвы
        if info.get("battle_result"):
            # Explicit upgrade lines: compare BLUE unit names by slot before/after battle.
            current_blue_names = snapshot_blue_names_by_pos()
            upgrade_lines = []
            for pos in sorted(current_blue_names.keys()):
                old_name = prev_blue_names.get(pos, "")
                new_name = current_blue_names.get(pos, "")
                if not old_name or not new_name:
                    continue
                if old_name.lower() == "пусто":
                    continue
                if old_name != new_name:
                    upgrade_lines.append(f"UPGRADE: {old_name} -> {new_name} (pos{pos})")

            if upgrade_lines:
                print("\n=== UNIT UPGRADES ===")
                for line in upgrade_lines:
                    print(line)
                    battle_logs_buffer.append(line)

            prev_blue_names = current_blue_names

            if hasattr(env_base, "_get_blue_state"):
                blue_units = env_base._get_blue_state()
            else:
                blue_units = getattr(env_base, "blue_team_state", []) or []
            print("\n=== ОПЫТ ЖИВЫХ ЮНИТОВ ГЕРОЯ ===")
            for unit in blue_units:
                hp = float(unit.get("health", 0) or unit.get("hp", 0) or 0)
                if hp <= 0:
                    continue
                name = unit.get("name", "unknown")
                pos = unit.get("position", "?")
                exp_kill = unit.get("exp_kill", 0)
                exp_required = unit.get("exp_required", 0)
                exp_current = unit.get("exp_current", 0)
                print(
                    f"[HERO pos{pos} {name}] exp_kill={exp_kill} "
                    f"exp_required={exp_required} exp_current={exp_current}"
                )

    try:
        scripted_bot_episode_kills = int(
            getattr(env_base, "scripted_capital_bot_enemies_defeated", 0) or 0
        )
    except (TypeError, ValueError):
        scripted_bot_episode_kills = 0
    print(
        f"\nИтог эпизода — побед бота столицы над отрядами врагов: "
        f"{scripted_bot_episode_kills}"
    )

    # Финальная отрисовка
    (
        heal_left,
        max_heal,
        healing_left,
        max_healing,
        revive_left,
        max_revive,
    ) = get_bottle_counters()
    hero_name, hero_level, hero_abilities = get_travel_hero_panel()
    grid_viz.draw_grid(
        agent_pos=env_base.grid_env.agent_pos,
        enemy_positions=env_base.grid_env.enemy_positions,
        enemies_alive=env_base.grid_env.enemies_alive,
        visited_cells=env_base.grid_env.visited_cells,
        title=f"CAMPAIGN {'VICTORY' if battles_won == total_enemies else 'ENDED'} | Reward: {total_reward:.2f}",
        castle_heal_tiles=getattr(env_base, "castle_heal_tiles", None),
        legions_territory_tiles=getattr(env_base, "legions_territory_tiles", None),
        empire_territory_tiles=getattr(env_base, "empire_territory_tiles", None),
        obstacle_tiles=getattr(env_base.grid_env, "obstacle_positions", None),
        merchant_positions=getattr(env_base.grid_env, "merchant_positions", None),
        merchant_site_anchors=getattr(env_base, "merchant_site_anchors", None),
        spell_shop_positions=getattr(env_base.grid_env, "spell_shop_positions", None),
        spell_shop_site_anchors=getattr(env_base, "spell_shop_site_anchors", None),
        mercenary_positions=getattr(env_base.grid_env, "mercenary_positions", None),
        mercenary_site_anchors=getattr(env_base, "mercenary_site_anchors", None),
        trainer_positions=getattr(env_base.grid_env, "trainer_positions", None),
        trainer_site_anchors=getattr(env_base, "trainer_site_anchors", None),
        mana_sources=getattr(env_base.grid_env, "mana_sources", None),
        mana_totals=env_base._current_mana_totals(),
        mana_income_per_turn=getattr(env_base, "mana_income_per_turn", None),
        turns=env_base.turns,
        gold=env_base.gold,
        steps=env_base.moves,
        heal_bottles_left=heal_left,
        max_heal_bottles=max_heal,
        healing_bottles_left=healing_left,
        max_healing_bottles=max_healing,
        revive_bottles_left=revive_left,
        max_revive_bottles=max_revive,
        built_buildings=env_base.get_built_building_names(),
        learned_spells=env_base.get_learned_spell_descriptions(),
        chest_positions=getattr(env_base, "chests", None),
        heroitems=getattr(env_base, "heroitems", None),
        equipped_hero_items=getattr(env_base, "equipped_hero_items", None),
        battle_items_equipped_total=getattr(env_base, "battle_items_equipped_total", 0),
        battle_items_used_total=getattr(env_base, "battle_items_used_total", 0),
        hero_name=hero_name,
        hero_level=hero_level,
        hero_abilities=hero_abilities,
        ruins_cleared=len(getattr(env_base, "cleared_ruin_enemy_ids", ())),
        ruins_total=len(getattr(env_base, "RUIN_REWARD_BY_ENEMY_ID", {})),
        last_ruin_reward=getattr(env_base, "last_ruin_reward", None),
        scripted_bot=env_base._scripted_capital_bot_info()["scripted_capital_bot"],
    )

    # Закрываем визуализатор боя если открыт
    if battle_viz:
        battle_viz.close()

    print("\nНажмите Enter для закрытия...")
    input()
    grid_viz.close()


def find_latest_model(base_dir: str = "campaign_models") -> str:
    """Находит последнюю обученную модель."""
    if not os.path.exists(base_dir):
        return None

    # Ищем все подпапки
    runs = []
    for d in os.listdir(base_dir):
        run_dir = os.path.join(base_dir, d)
        if os.path.isdir(run_dir):
            final_model = os.path.join(run_dir, "final_model.zip")
            if os.path.exists(final_model):
                runs.append((run_dir, os.path.getmtime(final_model)))

    if not runs:
        return None

    # Возвращаем самую новую
    latest = max(runs, key=lambda x: x[1])
    return os.path.join(latest[0], "final_model.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Campaign Agent")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model .zip file",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay between steps (seconds)",
    )
    parser.add_argument(
        "--battle-speed",
        type=float,
        default=1.0,
        help="Battle animation speed multiplier (>1 faster, <1 slower)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (default: stochastic)",
    )
    parser.add_argument(
        "--map-png",
        type=str,
        default=_existing_path_or_none(DEFAULT_MAP_PNG_PATH),
        help="Path to campaign background PNG. Defaults to outputs/a_return_to_simpler_times_entries.png if present.",
    )
    parser.add_argument(
        "--scenario-path",
        type=str,
        default=_existing_path_or_none(DEFAULT_SCENARIO_PATH),
        help="Path to scenario .sg used to highlight capitals and villages.",
    )
    args = parser.parse_args()

    # Ищем модель
    model_path = args.model
    if not model_path:
        model_path = find_latest_model()
        if model_path:
            print(f"Найдена последняя модель: {model_path}")

    run_campaign_visualization(
        model_path,
        delay=args.delay,
        battle_speed=args.battle_speed,
        deterministic=args.deterministic,
        map_png_path=args.map_png,
        scenario_path=args.scenario_path,
    )
