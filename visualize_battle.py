#!/usr/bin/env python3
"""
Визуализация боя с обученным агентом.
Этот файл запускает визуализацию боя после обучения модели.
"""
import sys
import os
sys.path.append('.')

from stable_baselines3 import PPO
from lesssimpleenv import BattleEnv
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch
import re

def visualize_trained_agent():
    """Визуализировать бой с обученным агентом."""

    print("=== ВИЗУАЛИЗАЦИЯ БОЯ С ОБУЧЕННЫМ АГЕНТОМ ===")

    # Загружаем обученную модель
    try:
        trained_model = PPO.load("ppo_blue_vs_red")
        print("✅ Обученная модель загружена успешно!")

        # Создаем среду для визуализации с логированием
        viz_env = BattleEnv(log_enabled=True, reward_illegal=-0.1)
        obs, info = viz_env.reset()

        # Собираем все строки логов для визуализации
        all_logs = []
        total_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            # Обученный агент выбирает действие
            action, _ = trained_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = viz_env.step(action)
            total_reward += reward

            # Собираем логи для визуализации
            new_lines = viz_env.pop_pretty_events()
            all_logs += new_lines

        print("\n=== ЭПИЗОД ЗАВЕРШЁН ===")
        print("Победитель:", viz_env.winner.upper(), "| Суммарная награда:", total_reward)

        # Импортируем необходимые библиотеки для визуализации
        import math

        # Параметры визуализации
        VISUAL_SPEED_MULT = 8.0
        FRAME_DELAY = 0.28

        # --- сетка поля
        RED_FRONT, RED_BACK   = [1,2,3],   [4,5,6]
        BLUE_FRONT, BLUE_BACK = [7,8,9],   [10,11,12]
        COL_X = {0: 0.18, 1: 0.50, 2: 0.82}
        Y_BLUE_BACK, Y_BLUE_FRONT = 0.88, 0.70
        Y_RED_FRONT,  Y_RED_BACK  = 0.30, 0.12
        SLOT_W, SLOT_H = 0.28, 0.13
        HP_H = 0.028

        # --- состояние для визуализации
        from lesssimpleenv import UNITS_RED, UNITS_BLUE
        state = {u["position"]: {"team": u["team"], "name": u["Name"], "hp": float(u["Health"]), "maxhp": float(u["Health"]),
                                "burn": float(u["Burn"]), "freeze": float(u["Freeze"]), "poison": float(u["Poison"]) }
                 for u in (UNITS_RED + UNITS_BLUE)}

        # Создаем фигуру для визуализации
        fig, ax = plt.subplots(figsize=(12, 8))

        # Определяем, работаем ли в Jupyter notebook
        try:
            from IPython.display import display
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
            elif u.get("Type") == "Lord":
                card_color = (0.90, 0.50, 0.20, 0.16)  # Оранжево-красный для Lord
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

            # HP бар
            hp_ratio = max(0, u["hp"] / u["maxhp"])
            hp_color = (0.2, 0.8, 0.2, 0.8) if hp_ratio > 0.6 else (0.8, 0.8, 0.2, 0.8) if hp_ratio > 0.3 else (0.8, 0.2, 0.2, 0.8)
            ax.add_patch(patches.FancyBboxPatch((x, y + SLOT_H*0.85), SLOT_W * hp_ratio, HP_H,
                                                boxstyle="round,pad=0.002", facecolor=hp_color, linewidth=0))
            hp_text = f"{int(u['hp'])}/{int(u['maxhp'])}"
            ax.text(x + SLOT_W/2, y + SLOT_H*0.85 + HP_H/2, hp_text,
                    fontsize=8, ha="center", va="center", color="black", fontweight="bold")

        def draw_board(arrows, headline: str, active_pos: int = None):
            ax.clear()
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
            ax.set_title(headline, fontsize=14, fontweight="bold", pad=20)

            # Рисуем всех юнитов
            for pos in [1,2,3,4,5,6,7,8,9,10,11,12]:
                if pos in state:
                    draw_unit(ax, pos, is_active=(pos == active_pos))

            # Рисуем стрелки
            for arrow in arrows:
                src_x, src_y = pos_center(arrow["src"])
                dst_x, dst_y = pos_center(arrow["dst"])
                if arrow.get("style") == "dashed":
                    style = (0, (5, 5))
                else:
                    style = 'solid'
                ax.add_patch(FancyArrowPatch((src_x, src_y), (dst_x, dst_y),
                                           color=arrow.get("team", "black"),
                                           linewidth=arrow.get("lw", 2.0),
                                           alpha=arrow.get("alpha", 1.0),
                                           linestyle=style,
                                           arrowstyle='->', mutation_scale=15))
                if "text" in arrow:
                    ax.text((src_x + dst_x)/2, (src_y + dst_y)/2, arrow["text"],
                           fontsize=9, ha="center", va="center", fontweight="bold")

            if IS_NOTEBOOK:
                if handle is not None:
                    handle.update(fig)
            else:
                fig.canvas.draw()
                try:
                    fig.canvas.flush_events()
                except Exception:
                    pass
            plt.pause(0.001)

        # Регулярные выражения для парсинга логов
        atk_re = re.compile(r"(\w+) (\w+)#(\d+) → (\w+) (\w+)#(\d+): урон (\d+) \((\d+)→(\d+)\)")
        status_damage_re = re.compile(r"(\w+) (\w+)#(\d+): урон от эффектов (\d+)")
        blue_action_re = re.compile(r"BLUE (\w+)#(\d+) → (\w+)#(\d+)")
        kill_re = re.compile(r"(\w+) (\w+)#(\d+) выведен из строя")
        vict_re = re.compile(r"ПОБЕДИТЕЛЬ: (\w+)")
        status_clear_re = re.compile(r"(\w+) (\w+)#(\d+): эффекты сняты: (.+)")

        # Парсим логи и создаем анимацию
        arrows_now = []
        current_actor_pos = None

        for line in all_logs:
            # Очищаем стрелки на новой инициативе
            if "ИНИЦИАТИВА" in line:
                arrows_now = []
                current_actor_pos = None

            m = atk_re.match(line)
            if m:
                atk_team = m.group(1); atk_pos = int(m.group(3))
                vic_pos = int(m.group(6)); dmg = int(m.group(7)); after = int(m.group(9))
                if vic_pos in state: state[vic_pos]["hp"] = after
                current_actor_pos = atk_pos
                arrows_now.append({"src": atk_pos, "dst": vic_pos, "team": atk_team, "text": f"-{dmg}"})
                headline = line
                draw_board(arrows_now, headline, active_pos=current_actor_pos)
                plt.pause(FRAME_DELAY * VISUAL_SPEED_MULT)
                continue

            m = status_damage_re.match(line)
            if m:
                team = m.group(1); pos = int(m.group(3)); dmg = int(m.group(4))
                if pos in state:
                    # Обновляем HP юнита после получения урона от эффектов
                    state[pos]["hp"] = max(0, state[pos]["hp"] - dmg)
                headline = line
                draw_board([], headline, active_pos=current_actor_pos)
                plt.pause((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2)
                continue

            m = blue_action_re.match(line)
            if m:
                src = int(m.group(2)); dst = int(m.group(4))
                last_selection = {"src": src, "dst": dst, "team": "BLUE"}
                current_actor_pos = src
                arrows_now.append({"src": src, "dst": dst, "team": "BLUE",
                                   "style":"dashed", "alpha":0.65, "lw":2.0})
                headline = line
                draw_board(arrows_now, headline, active_pos=current_actor_pos)
                plt.pause((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1)
                continue

            m = kill_re.match(line)
            if m:
                pos = int(m.group(3))
                if pos in state: state[pos]["hp"] = 0
                headline = line
                draw_board([], headline, active_pos=current_actor_pos)
                plt.pause((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1)
                continue

            m = status_clear_re.match(line)
            if m:
                pos = int(m.group(3))
                effects_str = m.group(4)
                if pos in state:
                    # Обнуляем эффекты в визуализации
                    if "Burn" in effects_str:
                        state[pos]["burn"] = 0
                    if "Freeze" in effects_str:
                        state[pos]["freeze"] = 0
                    if "Poison" in effects_str:
                        state[pos]["poison"] = 0
                headline = line
                draw_board([], headline, active_pos=current_actor_pos)
                plt.pause((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2)
                continue

            if vict_re.match(line) or line.startswith("— ") or "RED ход" in line:
                headline = line
                draw_board([], headline, active_pos=current_actor_pos)
                plt.pause((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1)
                continue

            # Прочие строки
            draw_board([], line, active_pos=current_actor_pos)
            plt.pause((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2)

        draw_board([], "Конец эпизода - обученный агент победил!", active_pos=None)
        plt.show()
        print("✅ Визуализация боя с обученным агентом завершена!")

    except FileNotFoundError:
        print("❌ Обученная модель не найдена. Запустите обучение сначала.")
        print("💡 Для обучения раскомментируйте строку: model.learn(total_timesteps=TOTAL_STEPS, callback=CallbackList(callbacks))")
        print("💡 в файле lesssimpleenv.py")
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели или визуализации: {e}")

if __name__ == "__main__":
    visualize_trained_agent()
