#!/usr/bin/env python3
"""
Демонстрация битвы с магами, воинами и лучниками
Визуализированная битва с ускоренной анимацией
"""

import os
import sys
import time
import math
import random
import re
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch

# Импортируем среду
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lesssimpleenv import BattleEnv

def run_battle_showcase():
    """Запустить демонстрацию битвы с визуализацией"""

    print("=== БИТВА МАГОВ, ВОИНОВ И ЛУЧНИКОВ ===")
    print("[MAGE] Маги атакуют всех вражеских юнитов")
    print("[WARRIOR] Воины атакуют только ближайших противников")
    print("[ARCHER] Лучники атакуют по одной цели")
    print("")

    # Создаем среду с разнообразными юнитами
    env = BattleEnv(log_enabled=True, reward_illegal=-0.1)
    obs, info = env.reset()

    print("Состав команд:")
    for unit in env.combined:
        unit_type = unit.get("Type", "Unknown")
        if unit_type == "Mage":
            icon = "[MAGE]"
        elif unit_type == "Warrior":
            icon = "[WARRIOR]"
        else:
            icon = "[ARCHER]"
        print(f"  {icon} {unit['team'].upper()} {unit['Name']}#{unit['position']} ({unit_type}, DMG: {unit['Damage']}, HP: {unit['Health']})")

    print("\n=== НАЧИНАЕМ БИТВУ! ===")

    # Собираем логи
    all_logs = []
    all_logs += env.pop_pretty_events()

    # Имитируем бой с ускоренной визуализацией
    step = 0
    max_steps = 50  # Ограничим шаги

    while step < max_steps:
        step += 1

        attacker_pos = env.current_blue_attacker_pos
        if attacker_pos is None:
            env._advance_until_blue_turn()
            if env.winner is not None:
                break
            attacker_pos = env.current_blue_attacker_pos

        if attacker_pos is not None:
            attacker = env._unit_by_position(attacker_pos)

            # Имитируем случайное действие
            action = random.randint(0, 5)
            target_pos = [1, 2, 3, 4, 5, 6][action]

            obs, reward, terminated, truncated, info = env.step(action)

            # Получаем логи
            new_logs = env.pop_pretty_events()
            all_logs += new_logs

            if terminated:
                break

    print(f"\nБой завершен после {step} шагов!")
    print(f"Победитель: {env.winner.upper()}")

    # ============== УСКОРЕННАЯ ГРАФИЧЕСКАЯ ВИЗУАЛИЗАЦИЯ ==============
    print("\n🎬 Запускаем ускоренную графическую визуализацию...")
    visualize_battle_graphical(all_logs)

def visualize_battle_console(all_logs):
    """Консольная визуализация битвы с ускоренной анимацией"""

    print("\n🎬 Запускаем ускоренную консольную визуализацию...")

    # Параметры визуализации (ускоренная)
    VISUAL_SPEED_MULT = 20.0  # Увеличено для ускорения
    FRAME_DELAY = 0.1         # Уменьшено для ускорения

    # --- регулярные выражения (для парсинга логов)
    atk_re         = re.compile(r'^(RED|BLUE)\s+[^#]+#(\d+)\s+→\s+(RED|BLUE)\s+[^#]+#(\d+):\s+(\d+)\s+\((\d+)→(\d+)\)')
    kill_re        = re.compile(r'^✖\s+(RED|BLUE)\s+[^#]+#(\d+)\s+выведен из строя\.')
    vict_re        = re.compile(r'^🏆 Победа (RED|BLUE)!')
    blue_turn_re   = re.compile(r'^Ход BLUE:\s+[^#]+#(\d+)')
    red_turn_re    = re.compile(r'^RED ход:\s+[^#]+#(\d+)')
    blue_action_re = re.compile(r'^BLUE действие:\s+[^#]+#(\d+)\s+→\s+pos(\d+)')
    mage_attack_re = re.compile(r'^🔮\s+(RED|BLUE)\s+[^#]+#(\d+)\s+\(Маг\)\s+атакует всех вражеских юнитов!')
    warrior_attack_re = re.compile(r'^⚔️\s+(RED|BLUE)\s+[^#]+#(\d+)\s+\(Воин\)\s+атакует pos(\d+)!')

    # Фильтруем только важные события для ускорения
    important_lines = []
    for line in all_logs:
        if (atk_re.match(line) or kill_re.match(line) or vict_re.match(line) or
            blue_turn_re.match(line) or red_turn_re.match(line) or blue_action_re.match(line) or
            mage_attack_re.match(line) or warrior_attack_re.match(line) or
            line.startswith("— ") or "ход" in line or "атакует" in line):
            important_lines.append(line)

    print(f"\nПоказываем {len(important_lines)} ключевых событий из {len(all_logs)} общих...")

    # --- проигрываем логи покадрово
    for i, line in enumerate(important_lines):
        print(f"\n[ШАГ {i+1}] {line}")
        time.sleep(FRAME_DELAY * VISUAL_SPEED_MULT / len(important_lines))

    print("\n✅ Визуализация завершена!")
    print("🎮 Было продемонстрировано:")
    print("  [MAGE] Маги атакуют всех врагов")
    print("  [WARRIOR] Воины атакуют только ближайших")
    print("  [ARCHER] Лучники фокусируются на одной цели")
    print("  [SHIELD] Системы защиты и сопротивления")
    print("  [DEATH] Прогрессивное уничтожение юнитов")

def visualize_battle_graphical(all_logs):
    """Графическая визуализация битвы с ускоренной анимацией"""

    print("\n🎬 Запускаем ускоренную графическую визуализацию...")

    # Параметры визуализации (ускоренная)
    VISUAL_SPEED_MULT = 15.0  # Увеличено для ускорения
    FRAME_DELAY = 0.15        # Уменьшено для ускорения

    # Получаем исходные юниты из глобальных переменных среды
    from lesssimpleenv import UNITS_RED, UNITS_BLUE

    # --- сетка поля
    RED_FRONT, RED_BACK   = [1,2,3],   [4,5,6]
    BLUE_FRONT, BLUE_BACK = [7,8,9],   [10,11,12]
    COL_X = {0: 0.18, 1: 0.50, 2: 0.82}
    Y_BLUE_BACK, Y_BLUE_FRONT = 0.88, 0.70
    Y_RED_FRONT,  Y_RED_BACK  = 0.30, 0.12
    SLOT_W, SLOT_H = 0.28, 0.13
    HP_H = 0.028

    # --- состояние (имена и стартовые HP для визуализации)
    state = {u["position"]: {"team": u["team"], "name": u["Name"], "hp": float(u["Health"]), "maxhp": float(u["Health"]), "type": u.get("Type", "Unknown")}
             for u in (UNITS_RED + UNITS_BLUE)}

    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.ioff()  # Отключаем интерактивный режим

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
        if pos not in state:
            return
        u = state[pos]; x, y = pos_to_xy(pos)
        # Скрываем задний ряд мёртвых (чтобы не загромождать)
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
        unit_type = u.get("type", "Unknown")
        type_display = f"({unit_type})"
        ax.text(x + 0.012, y + SLOT_H*0.78, name,
                fontsize=10, fontweight="bold", ha="left", va="center", color="black")
        ax.text(x + 0.012, y + SLOT_H*0.78 - 0.03, type_display,
                fontsize=8, ha="left", va="center", color=(0.4, 0.4, 0.4))

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

    # --- регулярные выражения (для парсинга логов)
    atk_re         = re.compile(r'^(RED|BLUE)\s+[^#]+#(\d+)\s+→\s+(RED|BLUE)\s+[^#]+#(\d+):\s+(\d+)\s+\((\d+)→(\d+)\)')
    kill_re        = re.compile(r'^✖\s+(RED|BLUE)\s+[^#]+#(\d+)\s+выведен из строя\.')
    vict_re        = re.compile(r'^🏆 Победа (RED|BLUE)!')
    blue_turn_re   = re.compile(r'^Ход BLUE:\s+[^#]+#(\d+)')
    red_turn_re    = re.compile(r'^RED ход:\s+[^#]+#(\d+)')
    blue_action_re = re.compile(r'^BLUE действие:\s+[^#]+#(\d+)\s+→\s+pos(\d+)')

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
        plt.pause(0.001)

    # --- проигрываем логи покадрово
    current_actor_pos = None
    draw_board(headline="Битва магов, воинов и лучников!", active_pos=current_actor_pos)
    time.sleep(FRAME_DELAY * 2)  # Пауза в начале

    last_selection = None
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
            atk_team = m.group(1); atk_pos = int(m.group(2))
            vic_pos = int(m.group(4)); dmg = int(m.group(5)); after = int(m.group(7))
            if vic_pos in state:
                state[vic_pos]["hp"] = after
                # Также обновляем тип юнита на всякий случай
                state[vic_pos]["type"] = state[vic_pos].get("type", "Unknown")
            current_actor_pos = atk_pos
            arrows_now.append({"src": atk_pos, "dst": vic_pos, "team": atk_team, "text": f"-{dmg}"})
            headline = line
            draw_board(arrows_now, headline, active_pos=current_actor_pos)
            time.sleep(FRAME_DELAY * VISUAL_SPEED_MULT)
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

        m = kill_re.match(line)
        if m:
            pos = int(m.group(2))
            if pos in state:
                state[pos]["hp"] = 0
                # Тип юнита сохраняется даже после смерти
            headline = line
            draw_board([], headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1)
            continue

        if vict_re.match(line) or line.startswith("— ") or "RED ход" in line:
            headline = line
            draw_board([], headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1)
            continue

        # Прочие строки (включая атаки магов и воинов)
        draw_board([], line, active_pos=current_actor_pos)
        time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2)

    # Финальный кадр
    draw_board([], "[WINNER] Битва завершена! [WINNER]", active_pos=None)
    time.sleep(FRAME_DELAY * 3)  # Длинная пауза в конце

    print("\n✅ Графическая визуализация завершена!")
    print("🎮 Было продемонстрировано:")
    print("  [MAGE] Маги атакуют всех врагов")
    print("  [WARRIOR] Воины атакуют только ближайших")
    print("  [ARCHER] Лучники фокусируются на одной цели")
    print("  [SHIELD] Системы защиты и сопротивления")
    print("  [DEATH] Прогрессивное уничтожение юнитов")

if __name__ == "__main__":
    run_battle_showcase()
