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

from campaign_env import CampaignEnv
from enemy_configs import ENEMY_DESCRIPTIONS
from battle_env import (
    TARGET_POSITIONS,
    DEFEND_ACTION_INDEX,
    WAIT_ACTION_INDEX,
    RUN_AWAY_ACTION_INDEX,
    DEFEND_ARMOR_BONUS,
)
from console_encoding import setup_utf8_console


setup_utf8_console()


def mask_fn(env: CampaignEnv) -> np.ndarray:
    return env.compute_action_mask()


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
    SLOT_W, SLOT_H = 0.28, 0.13
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
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
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
        ax.text(x + 0.012, y + self.SLOT_H * 0.78, (u["name"] or "")[:22], fontsize=10, fontweight="bold", ha="left", va="center", color="black")
        if u.get("acc2", 0) > 0:
            ax.text(x + self.SLOT_W - 0.012, y + self.SLOT_H * 0.78, f"{int(u['acc2'])}%", fontsize=9, ha="right", va="center", color="black")
        hp_x, hp_y = x + 0.012, y + self.SLOT_H * 0.10
        hp_w = self.SLOT_W - 0.024
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
        ax.text(x + 0.012, y + height * 0.82, (uf["name"] or "")[:22], fontsize=11, fontweight="bold", ha="left", va="center", color="black")
        if uf.get("acc2", 0) > 0:
            ax.text(x + width - 0.012, y + height * 0.82, f"{int(uf['acc2'])}%", fontsize=10, ha="right", va="center", color="black")
        hp_w = width - 0.024
        hp_x, hp_y = x + 0.012, y + height * 0.10
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
            self.ax.text(0.5, 0.985, headline[:120], fontsize=12, fontweight="bold", ha="center", va="top", color="black")
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
            src_team, src_pos, dst_team, dst_pos, dmg, after = m.group(1), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5)), int(m.group(7))
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
            self.draw_board(headline=f"?? {line}", active_pos=None)
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
    COLOR_ENEMY_ALIVE = (0.9, 0.3, 0.3, 0.8)
    COLOR_ENEMY_DEAD = (0.5, 0.5, 0.5, 0.4)
    COLOR_VISITED = (0.7, 0.85, 0.7, 0.3)
    COLOR_UNVISITED = (0.95, 0.95, 0.95, 1.0)
    COLOR_PATH = (0.3, 0.6, 0.9, 0.6)
    COLOR_CASTLE = (1.0, 0.92, 0.2, 0.55)
    CASTLE_POS = (1, 1)

    def __init__(self, grid_size: int = 40, cell_size: float = 0.8):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.path_history = []

        # Создаём фигуру
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.ion()
        self.fig.show()

    def draw_grid(
        self,
        agent_pos: tuple,
        enemy_positions: dict,
        enemies_alive: dict,
        visited_cells: set,
        title: str = "",
        highlight_battle: int = None,
        castle_heal_tiles: list | tuple | None = None,
        turns: int = 0,
        gold: float = 0.0,
        steps: int = 0,
        heal_bottles_left: int = 0,
        max_heal_bottles: int = 0,
        revive_bottles_left: int = 0,
        max_revive_bottles: int = 0,
        built_buildings: list = None,
        hero_name: str = "",
        hero_level: int = 0,
        hero_abilities: list | None = None,
    ):
        """Отрисовка карты."""
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect("equal")
        self.ax.invert_yaxis()  # Y растёт вниз

        # Сетка
        for i in range(self.grid_size + 1):
            self.ax.axhline(y=i - 0.5, color="gray", linewidth=0.5, alpha=0.5)
            self.ax.axvline(x=i - 0.5, color="gray", linewidth=0.5, alpha=0.5)

        # Посещённые клетки
        for (x, y) in visited_cells:
            rect = patches.Rectangle(
                (x - 0.45, y - 0.45), 0.9, 0.9,
                facecolor=self.COLOR_VISITED,
                edgecolor="none",
            )
            self.ax.add_patch(rect)

        # Heal tiles (castle + extra heal cells) — highlight in yellow.
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
            if (cx, cy) in seen_tiles:
                continue
            seen_tiles.add((cx, cy))
            castle_rect = patches.Rectangle(
                (cx - 0.45, cy - 0.45), 0.9, 0.9,
                facecolor=self.COLOR_CASTLE,
                edgecolor=(0.9, 0.6, 0.0),
                linewidth=2.0,
            )
            self.ax.add_patch(castle_rect)
            self.ax.text(
                cx, cy, "C",
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color=(0.35, 0.25, 0.0),
            )

        # Путь агента
        if len(self.path_history) > 1:
            for i in range(len(self.path_history) - 1):
                x1, y1 = self.path_history[i]
                x2, y2 = self.path_history[i + 1]
                self.ax.plot(
                    [x1, x2], [y1, y2],
                    color=self.COLOR_PATH,
                    linewidth=2,
                    alpha=0.7,
                )

        # Враги
        for enemy_id, (ex, ey) in enemy_positions.items():
            is_alive = enemies_alive.get(enemy_id, False)
            color = self.COLOR_ENEMY_ALIVE if is_alive else self.COLOR_ENEMY_DEAD

            # Подсветка текущего боя
            if highlight_battle == enemy_id:
                circle = plt.Circle(
                    (ex, ey), 0.55,
                    facecolor="none",
                    edgecolor=(1.0, 0.8, 0.0),
                    linewidth=4,
                )
                self.ax.add_patch(circle)

            circle = plt.Circle((ex, ey), 0.35, facecolor=color, edgecolor="black", linewidth=1.5)
            self.ax.add_patch(circle)

            # Номер врага
            status = "OK" if not is_alive else ""
            self.ax.text(
                ex, ey, f"E{enemy_id}{status}",
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if is_alive else "gray",
            )

        # Агент
        agent_circle = plt.Circle(
            agent_pos, 0.3,
            facecolor=self.COLOR_AGENT,
            edgecolor="darkblue",
            linewidth=2,
        )
        self.ax.add_patch(agent_circle)
        self.ax.text(
            agent_pos[0], agent_pos[1], "A",
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color="white",
        )

        # Заголовок
        self.ax.set_title(title, fontsize=14, fontweight="bold")

        # Ход, золото, шаги и постройки справа от грида
        info_lines = [
            f"Ход: {turns}",
            f"Золото: {gold:g}",
            f"Шаги: {steps}",
            f"Леч. бутылки: {heal_bottles_left}/{max_heal_bottles}",
            f"Воскр. бутылки: {revive_bottles_left}/{max_revive_bottles}",
        ]
        built_buildings = built_buildings or []
        if built_buildings:
            info_lines.append("Постройки:")
            info_lines.extend(f"- {name}" for name in built_buildings)
        else:
            info_lines.append("Постройки: нет")
        hero_name = str(hero_name or "").strip()
        hero_abilities = list(hero_abilities or [])
        if hero_name:
            info_lines.append(f"Герой: {hero_name} (ур. {hero_level})")
            info_lines.append("Способности героя:")
            if hero_abilities:
                info_lines.extend(f"- {ability}" for ability in hero_abilities)
            else:
                info_lines.append("- нет")
        info_text = "\n".join(info_lines)
        self.ax.text(
            1.02,
            0.98,
            info_text,
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="black",
            clip_on=False,
        )

        # Легенда
        legend_text = f"Visited: {len(visited_cells)}/{self.grid_size**2}"
        alive_count = sum(1 for v in enemies_alive.values() if v)
        total_enemies = len(enemies_alive)
        defeated_count = total_enemies - alive_count
        legend_text += f" | Enemies: {defeated_count}/{total_enemies} defeated"
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
    env_base = CampaignEnv(log_enabled=True, realcapital=2)
    env = ActionMasker(env_base, mask_fn)

    # Визуализаторы
    grid_viz = CampaignVisualizer(grid_size=env_base.grid_size)
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
        max_heal = int(getattr(env_base, "MAX_HEAL_BOTTLES", 0) or 0)
        max_revive = int(getattr(env_base, "MAX_REVIVE_BOTTLES", 0) or 0)
        heal_used = int(getattr(env_base, "heal_bottles_used", 0) or 0)
        revive_used = int(getattr(env_base, "revive_bottles_used", 0) or 0)
        heal_left = max(0, max_heal - heal_used)
        revive_left = max(0, max_revive - revive_used)
        return heal_left, max_heal, revive_left, max_revive

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

        action, _ = model.predict(obs_for_model, deterministic=True, action_masks=mask)

        # Логируем действие агента в режиме боя
        action_idx = int(action)
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
            else:
                chosen_line = f"[STEP {step_count}] Агент выбирает action={action_idx}"
            print("\n" + chosen_line)
            battle_logs_buffer.append(chosen_line)

        # Выполняем шаг
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Определяем текущий режим
        new_mode = info.get("mode", "grid")

        # Визуализация в зависимости от режима
        if new_mode == "grid":
            agent_pos = env_base.grid_env.agent_pos
            grid_viz.add_to_path(agent_pos)

            # Заголовок
            action_names = ["↑", "↓", "←", "→", "↖", "↗", "↙", "↘", "⊙"]
            action_name = action_names[min(int(action), 8)]
            title = f"Step {step_count} | Action: {action_name} | Reward: {total_reward:.2f}"
            heal_left, max_heal, revive_left, max_revive = get_bottle_counters()
            hero_name, hero_level, hero_abilities = get_travel_hero_panel()

            grid_viz.draw_grid(
                agent_pos=agent_pos,
                enemy_positions=env_base.grid_env.enemy_positions,
                enemies_alive=env_base.grid_env.enemies_alive,
                visited_cells=env_base.grid_env.visited_cells,
                title=title,
                castle_heal_tiles=getattr(env_base, "castle_heal_tiles", None),
                turns=env_base.turns,
                gold=env_base.gold,
                steps=env_base.moves,
                heal_bottles_left=heal_left,
                max_heal_bottles=max_heal,
                revive_bottles_left=revive_left,
                max_revive_bottles=max_revive,
                built_buildings=env_base.get_built_building_names(),
                hero_name=hero_name,
                hero_level=hero_level,
                hero_abilities=hero_abilities,
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
                heal_left, max_heal, revive_left, max_revive = get_bottle_counters()
                hero_name, hero_level, hero_abilities = get_travel_hero_panel()

                grid_viz.draw_grid(
                    agent_pos=env_base.grid_env.agent_pos,
                    enemy_positions=env_base.grid_env.enemy_positions,
                    enemies_alive=env_base.grid_env.enemies_alive,
                    visited_cells=env_base.grid_env.visited_cells,
                    title=f"BATTLE vs Enemy {enemy_id}!",
                    highlight_battle=enemy_id,
                    castle_heal_tiles=getattr(env_base, "castle_heal_tiles", None),
                    turns=env_base.turns,
                    gold=env_base.gold,
                    steps=env_base.moves,
                    heal_bottles_left=heal_left,
                    max_heal_bottles=max_heal,
                    revive_bottles_left=revive_left,
                    max_revive_bottles=max_revive,
                    built_buildings=env_base.get_built_building_names(),
                    hero_name=hero_name,
                    hero_level=hero_level,
                    hero_abilities=hero_abilities,
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

    # Финальная отрисовка
    heal_left, max_heal, revive_left, max_revive = get_bottle_counters()
    hero_name, hero_level, hero_abilities = get_travel_hero_panel()
    grid_viz.draw_grid(
        agent_pos=env_base.grid_env.agent_pos,
        enemy_positions=env_base.grid_env.enemy_positions,
        enemies_alive=env_base.grid_env.enemies_alive,
        visited_cells=env_base.grid_env.visited_cells,
        title=f"CAMPAIGN {'VICTORY' if battles_won == total_enemies else 'ENDED'} | Reward: {total_reward:.2f}",
        castle_heal_tiles=getattr(env_base, "castle_heal_tiles", None),
        turns=env_base.turns,
        gold=env_base.gold,
        steps=env_base.moves,
        heal_bottles_left=heal_left,
        max_heal_bottles=max_heal,
        revive_bottles_left=revive_left,
        max_revive_bottles=max_revive,
        built_buildings=env_base.get_built_building_names(),
        hero_name=hero_name,
        hero_level=hero_level,
        hero_abilities=hero_abilities,
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
    )
