# ============================================================
# ТЕСТ МОДЕЛИ ИЗ ЧЕКПОИНТА: ЛОГИ + ВИЗУАЛИЗАЦИЯ (big/яд/поджог/Точность2)
# Промахи случайные: нигде не фиксируем seed.
# ============================================================

import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch

# Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
except Exception:  # pragma: no cover
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor


# --- загрузка среды, юнитов и констант из 1workingbattle.py ---
def _load_env_namespace() -> Dict[str, Any]:
    module_path = Path(__file__).with_name("1workingbattle.py")
    if not module_path.exists():
        raise FileNotFoundError(
            "Файл 1workingbattle.py не найден рядом с этим скриптом."
        )

    with module_path.open("r", encoding="utf-8") as fh:
        source = fh.read()

    split_token = "# ============================================================\n# ОБУЧЕНИЕ PPO"
    if split_token in source:
        source = source.split(split_token, 1)[0]

    namespace: Dict[str, Any] = {}
    exec(compile(source, str(module_path), "exec"), namespace)
    return namespace


_ENV = _load_env_namespace()

BattleEnv = _ENV["BattleEnv"]
TARGET_POSITIONS: List[int] = list(_ENV.get("TARGET_POSITIONS", list(range(1, 7))))
UNITS_RED = _ENV["UNITS_RED"]
UNITS_BLUE = _ENV["UNITS_BLUE"]


# --- путь к чекпоинту ---
def _resolve_checkpoint_path() -> Path:
    custom = os.environ.get("BATTLE_CKPT_PATH")
    if custom:
        candidate = Path(custom).expanduser()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Указанный чекпоинт не найден: {candidate}")

    root = Path("./checkpoints")
    if not root.exists():
        raise FileNotFoundError(
            "Каталог ./checkpoints не найден и переменная BATTLE_CKPT_PATH не задана"
        )

    candidates = sorted(
        (p for p in root.rglob("ppo_step_*.zip") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Не найдено файлов ppo_step_*.zip в ./checkpoints")

    chosen = candidates[0]
    print(f"[INFO] Используется чекпоинт: {chosen}")
    return chosen


CHECKPOINT_PATH = _resolve_checkpoint_path()

# --- загрузка модели и подготовка сред ---
eval_env = Monitor(BattleEnv(log_enabled=False))
ckpt_model = PPO.load(str(CHECKPOINT_PATH), env=eval_env, device="auto")

test_env_ckpt = BattleEnv(log_enabled=True)
obs, info = test_env_ckpt.reset()  # без seed → промахи/инициативы рандомны

# --- прогон с логами ---
all_logs_ckpt: List[str] = []
all_logs_ckpt += test_env_ckpt.pop_pretty_events()

done = False
total_reward = 0.0
step_i = 0
while not done:
    step_i += 1
    action, _ = ckpt_model.predict(obs, deterministic=True)

    target_pos = TARGET_POSITIONS[int(action)]
    chosen_line = f"[CKPT STEP {step_i}] Агент выбирает action={int(action)} → атака RED pos{target_pos}"
    print("\n" + chosen_line)
    all_logs_ckpt.append(chosen_line)

    obs, reward, terminated, truncated, info = test_env_ckpt.step(action)
    total_reward += reward
    done = terminated or truncated

    new_lines = test_env_ckpt.pop_pretty_events()
    all_logs_ckpt += new_lines
    for line in new_lines:
        print(line)

print("\n=== ЭПИЗОД (ЧЕКПОИНТ) ЗАВЕРШЁН ===")
print("Победитель:", test_env_ckpt.winner.upper(), "| Суммарная награда:", total_reward)

# ============================================================
# ВИЗУАЛИЗАЦИЯ ЛОГОВ (big-юниты, HP-бары, яд/поджог, Точность2)
# ============================================================

VISUALIZE = True
FRAME_DELAY = 0.28

if VISUALIZE:
    VISUAL_SPEED_MULT = 8.0

    # --- сетка поля
    RED_FRONT, RED_BACK = [1, 2, 3], [4, 5, 6]
    BLUE_FRONT, BLUE_BACK = [7, 8, 9], [10, 11, 12]
    COL_X = {0: 0.18, 1: 0.50, 2: 0.82}
    Y_BLUE_BACK, Y_BLUE_FRONT = 0.88, 0.70
    Y_RED_FRONT, Y_RED_BACK = 0.30, 0.12
    SLOT_W, SLOT_H = 0.28, 0.13
    HP_H = 0.028

    # --- состояние для отрисовки (берём стартовые HP из исходных юнитов)
    state: Dict[int, Dict[str, Any]] = {}

    def _suffix_by_type(t: str) -> str:
        return {
            "Mage": " (Mage)",
            "gargoil": " (Gargoil)",
            "Воин": " (воин)",
            "Demon": " (Demon)",
            "Death": " (Death)",
            "Archer": " (лучник)",
            "lord": " (Lord)",
            "Dead dragon": " (Dead dragon)",
        }.get(t, f" ({t})")

    for u in (UNITS_RED + UNITS_BLUE):
        t = u.get("Type", "Archer")
        start_hp = float(u["Здоровье"])
        state[u["position"]] = {
            "team": u["team"],
            "name": u["имя"] + _suffix_by_type(t),
            "hp": start_hp,
            "maxhp": start_hp,
            "stand": u["stand"],
            "type": t,
            "big": bool(u.get("big", False)),
            "acc2": float(u.get("Точность2", 0)),
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
        x = COL_X[col] - SLOT_W / 2
        if pos in BLUE_BACK:
            y = Y_BLUE_BACK - SLOT_H / 2
        elif pos in BLUE_FRONT:
            y = Y_BLUE_FRONT - SLOT_H / 2
        elif pos in RED_FRONT:
            y = Y_RED_FRONT - SLOT_H / 2
        else:
            y = Y_RED_BACK - SLOT_H / 2
        return x, y

    def pos_center(pos: int):
        x, y = pos_to_xy(pos)
        return x + SLOT_W / 2, y + SLOT_H / 2

    def draw_unit(ax, pos, is_active: bool = False):
        u = state[pos]
        x, y = pos_to_xy(pos)
        if pos in (RED_BACK + BLUE_BACK) and u["hp"] <= 0:
            return
        card_color = (
            (0.90, 0.30, 0.30, 0.16)
            if u["team"] == "red"
            else (0.30, 0.45, 0.90, 0.16)
        )
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                SLOT_W,
                SLOT_H,
                boxstyle="round,pad=0.010,rounding_size=0.016",
                linewidth=1.3,
                edgecolor="black",
                facecolor=card_color,
            )
        )
        if is_active:
            ax.add_patch(
                patches.FancyBboxPatch(
                    (x - 0.01, y - 0.01),
                    SLOT_W + 0.02,
                    SLOT_H + 0.02,
                    boxstyle="round,pad=0.012,rounding_size=0.018",
                    linewidth=3.0,
                    edgecolor=(1.0, 0.82, 0.10),
                    facecolor="none",
                    alpha=0.95,
                )
            )
        name = u["name"][:22]
        ax.text(
            x + 0.012,
            y + SLOT_H * 0.78,
            name,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
            color="black",
        )
        if u.get("acc2", 0) > 0:
            ax.text(
                x + SLOT_W - 0.012,
                y + SLOT_H * 0.78,
                f"☠{int(u['acc2'])}%",
                fontsize=9,
                ha="right",
                va="center",
                color="black",
            )

        hp_x, hp_y = x + 0.012, y + SLOT_H * 0.10
        hp_w = SLOT_W - 0.024
        ax.add_patch(
            patches.Rectangle((hp_x, hp_y), hp_w, HP_H, facecolor=(0.88, 0.88, 0.88), edgecolor="none")
        )
        frac = max(0.0, min(1.0, u["hp"] / u["maxhp"])) if u["maxhp"] > 1e-9 else 0.0
        ax.add_patch(
            patches.Rectangle((hp_x, hp_y), hp_w * frac, HP_H, facecolor=(0.15, 0.70, 0.25), edgecolor="none")
        )
        ax.text(
            hp_x + hp_w / 2,
            hp_y + HP_H / 2,
            f"{int(max(0, u['hp']))}/{int(u['maxhp'])}",
            fontsize=9,
            ha="center",
            va="center",
            color="black",
        )

        ax.text(
            x + SLOT_W / 2,
            y - 0.008,
            f"pos{pos}",
            fontsize=8,
            ha="center",
            va="top",
            color=(0.2, 0.2, 0.2),
        )

    def draw_big_unit(ax, front_pos: int, is_active: bool = False):
        back_pos = front_pos + 3
        uf = state[front_pos]
        x_front, y_front = pos_to_xy(front_pos)
        _, y_back = pos_to_xy(back_pos)
        x = x_front
        y = min(y_front, y_back)
        height = (max(y_front, y_back) + SLOT_H) - y
        width = SLOT_W
        if uf["hp"] <= 0:
            return
        card_color = (
            (0.90, 0.30, 0.30, 0.20)
            if uf["team"] == "red"
            else (0.30, 0.45, 0.90, 0.20)
        )
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.020",
                linewidth=1.6,
                edgecolor="black",
                facecolor=card_color,
            )
        )
        if is_active:
            ax.add_patch(
                patches.FancyBboxPatch(
                    (x - 0.012, y - 0.012),
                    width + 0.024,
                    height + 0.024,
                    boxstyle="round,pad=0.012,rounding_size=0.022",
                    linewidth=3.2,
                    edgecolor=(1.0, 0.82, 0.10),
                    facecolor="none",
                    alpha=0.95,
                )
            )
        name = uf["name"][:22]
        ax.text(
            x + 0.012,
            y + height * 0.82,
            name,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="center",
            color="black",
        )
        if uf.get("acc2", 0) > 0:
            ax.text(
                x + width - 0.012,
                y + height * 0.82,
                f"☠{int(uf['acc2'])}%",
                fontsize=10,
                ha="right",
                va="center",
                color="black",
            )

        hp_w = width - 0.024
        hp_x, hp_y = x + 0.012, y + height * 0.10
        ax.add_patch(
            patches.Rectangle((hp_x, hp_y), hp_w, HP_H, facecolor=(0.88, 0.88, 0.88), edgecolor="none")
        )
        frac = max(0.0, min(1.0, uf["hp"] / uf["maxhp"])) if uf["maxhp"] > 1e-9 else 0.0
        ax.add_patch(
            patches.Rectangle((hp_x, hp_y), hp_w * frac, HP_H, facecolor=(0.15, 0.70, 0.25), edgecolor="none")
        )
        ax.text(
            hp_x + hp_w / 2,
            hp_y + HP_H / 2,
            f"{int(max(0, uf['hp']))}/{int(uf['maxhp'])}",
            fontsize=9,
            ha="center",
            va="center",
            color="black",
        )
        ax.text(
            x + width / 2,
            y - 0.010,
            f"pos{front_pos}+{back_pos}",
            fontsize=8,
            ha="center",
            va="top",
            color=(0.2, 0.2, 0.2),
        )

    def draw_attack_arrow(
        ax,
        src_pos: int,
        dst_pos: int,
        team: str,
        text: str | None = None,
        style: str = "solid",
        color: tuple | None = None,
        alpha: float = 0.95,
        curve: float = 0.18,
        lw: float = 2.6,
    ):
        sx, sy = pos_center(src_pos)
        dx, dy = pos_center(dst_pos)
        vx, vy = dx - sx, dy - sy
        dist = math.hypot(vx, vy) + 1e-9
        pad = 0.06
        sx += vx / dist * pad
        sy += vy / dist * pad
        dx -= vx / dist * pad
        dy -= vy / dist * pad

        if color is None:
            color = (0.20, 0.35, 0.85) if team == "BLUE" else (0.85, 0.25, 0.25)

        con_style = f"arc3,rad={curve}" if abs(vx) > 0.01 and abs(vy) > 0.01 else "arc3,rad=0.0"
        arrow = FancyArrowPatch(
            (sx, sy),
            (dx, dy),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            linestyle="--" if style == "dashed" else "solid",
            color=color,
            alpha=alpha,
            connectionstyle=con_style,
        )
        ax.add_patch(arrow)

        if text:
            mx, my = (sx + dx) / 2, (sy + dy) / 2
            ax.text(
                mx,
                my + 0.03,
                text,
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                color=color,
            )

    # --- регексы (учитываем «цель с мин. HP pos...» и события статусов)
    atk_re = re.compile(r"^(RED|BLUE)\s+[^#]+#(\d+)\s+→\s+(RED|BLUE)\s+[^#]+#(\d+):\s+(\d+)\s+\((\d+)→(\d+)\)")
    kill_re = re.compile(r"^✖\s+(RED|BLUE)\s+[^#]+#(\d+)\s+выведен из строя\.")
    vict_re = re.compile(r"^🏆 Победа (RED|BLUE)!")
    blue_turn_re = re.compile(r"^Ход BLUE:\s+[^#]+#(\d+)")
    red_turn_re = re.compile(r"^RED ход:\s+[^#]+#(\d+)")
    blue_action_re = re.compile(r"^BLUE действие:\s+[^#]+#(\d+)\s+→\s+pos(\d+)")
    red_target_re = re.compile(r"^RED ход:\s+[^#]+#(\d+).+цель.*pos(\d+)")
    mage_banner_re = re.compile(r"^\w+\s+[^#]+#(\d+)\s+\((?:Mage|Dead dragon)\).+массов")
    blue_cant_re = re.compile(r"^BLUE\s+[^#]+#(\d+).+не может достать pos(\d+)")
    poison_tick_re = re.compile(r"^☠ Яд поражает (RED|BLUE)\s+[^#]+#(\d+):\s+(\d+)\s+\((\d+)→(\d+)\)")
    burn_tick_re = re.compile(r"^🔥 Поджог поражает (RED|BLUE)\s+[^#]+#(\d+):\s+(\d+)\s+\((\d+)→(\d+)\)")
    immune_dmg_re = re.compile(r"^🛡 Иммунитет к урону")
    immune_stat_re = re.compile(r"^🛡 Иммунитет к эффекту")
    resist_re = re.compile(r"^🧿 Стойкость")
    miss_re = re.compile(r"^💨 Промах:\s+(RED|BLUE)\s+[^#]+#(\d+)\s+по\s+(RED|BLUE)\s+[^#]+#(\d+)\.")
    poison_roll_re = re.compile(r"^🧪 Шанс отравления \d+% — (успех|неудача)")

    def draw_board(
        arrows: List[Dict[str, Any]] | None = None,
        headline: str = "",
        active_pos: int | None = None,
    ):
        ax.cla()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.plot([0.02, 0.98], [0.50, 0.50], color=(0.6, 0.6, 0.6), lw=1.2, ls="--", alpha=0.7)
        ax.text(0.01, 0.96, "BLUE (top)", color=(0.2, 0.35, 0.8), fontsize=12, fontweight="bold", ha="left")
        ax.text(0.01, 0.04, "RED (bottom)", color=(0.8, 0.25, 0.25), fontsize=12, fontweight="bold", ha="left")

        arrows = arrows or []
        drawn_big_pairs = set()

        for front_pos in (BLUE_FRONT + RED_FRONT):
            if front_pos in state and state[front_pos].get("big", False):
                draw_big_unit(ax, front_pos, is_active=(active_pos == front_pos))
                drawn_big_pairs.add((front_pos, front_pos + 3))

        for pos in (BLUE_BACK + BLUE_FRONT + RED_FRONT + RED_BACK):
            if (pos - 3, pos) in drawn_big_pairs or (pos, pos + 3) in drawn_big_pairs:
                continue
            draw_unit(ax, pos, is_active=(active_pos == pos))

        for a in arrows:
            draw_attack_arrow(
                ax,
                a["src"],
                a["dst"],
                a["team"],
                text=a.get("text"),
                style=a.get("style", "solid"),
                color=a.get("color"),
                alpha=a.get("alpha", 0.95),
                curve=a.get("curve", 0.18),
                lw=a.get("lw", 2.6),
            )

        if headline:
            ax.text(0.5, 0.52, headline[:140], fontsize=12, ha="center", va="bottom", color=(0.15, 0.15, 0.15))

        fig.canvas.draw()
        if handle is not None and hasattr(handle, "update"):
            handle.update(fig)
        else:
            try:
                fig.canvas.flush_events()
            except Exception:
                pass
            plt.pause(0.001)

    current_actor_pos = None
    draw_board(headline="Начало боя (чекпоинт)", active_pos=current_actor_pos)

    last_selection: Dict[str, Any] | None = None
    for line in all_logs_ckpt:
        arrows_now: List[Dict[str, Any]] = []
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
            atk_pos = int(m.group(2))
            vic_pos = int(m.group(4))
            dmg = int(m.group(5))
            after = int(m.group(7))
            if vic_pos in state:
                state[vic_pos]["hp"] = after
            current_actor_pos = atk_pos
            arrows_now.append({"src": atk_pos, "dst": vic_pos, "team": atk_team, "text": f"-{dmg}"})
            headline = line
            draw_board(arrows_now, headline, active_pos=current_actor_pos)
            time.sleep(FRAME_DELAY * VISUAL_SPEED_MULT)
            continue

        m = poison_tick_re.match(line)
        if m:
            vic_pos = int(m.group(2))
            after = int(m.group(5))
            if vic_pos in state:
                state[vic_pos]["hp"] = after
            headline = line
            draw_board([], headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2)
            continue

        m = burn_tick_re.match(line)
        if m:
            vic_pos = int(m.group(2))
            after = int(m.group(5))
            if vic_pos in state:
                state[vic_pos]["hp"] = after
            headline = line
            draw_board([], headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2)
            continue

        m = blue_action_re.match(line)
        if m:
            src = int(m.group(1))
            dst = int(m.group(2))
            last_selection = {"src": src, "dst": dst, "team": "BLUE"}
            current_actor_pos = src
            arrows_now.append(
                {
                    "src": src,
                    "dst": dst,
                    "team": "BLUE",
                    "style": "dashed",
                    "alpha": 0.65,
                    "lw": 2.0,
                }
            )
            headline = line
            draw_board(arrows_now, headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1)
            continue

        m = red_target_re.match(line)
        if m:
            src = int(m.group(1))
            dst = int(m.group(2))
            last_selection = {"src": src, "dst": dst, "team": "RED"}
            current_actor_pos = src
            arrows_now.append(
                {
                    "src": src,
                    "dst": dst,
                    "team": "RED",
                    "style": "dashed",
                    "alpha": 0.65,
                    "lw": 2.0,
                }
            )
            headline = line
            draw_board(arrows_now, headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1)
            continue

        if immune_dmg_re.match(line) or immune_stat_re.match(line) or resist_re.match(line):
            if last_selection:
                arrows_now.append(
                    {
                        "src": last_selection["src"],
                        "dst": last_selection["dst"],
                        "team": last_selection["team"],
                        "color": (0.45, 0.45, 0.45),
                        "style": "solid",
                        "alpha": 0.9,
                        "text": "RESIST",
                    }
                )
                current_actor_pos = last_selection["src"]
            headline = line
            draw_board(arrows_now, headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2)
            continue

        m = blue_cant_re.match(line)
        if m:
            src = int(m.group(1))
            dst = int(m.group(2))
            arrows_now.append(
                {
                    "src": src,
                    "dst": dst,
                    "team": "BLUE",
                    "color": (0.5, 0.5, 0.5),
                    "style": "dashed",
                    "alpha": 0.9,
                    "text": "недосягаемо",
                }
            )
            current_actor_pos = src
            headline = line
            draw_board(arrows_now, headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2)
            continue

        m = miss_re.match(line)
        if m:
            atk_team = m.group(1)
            atk_pos = int(m.group(2))
            vic_pos = int(m.group(4))
            current_actor_pos = atk_pos
            arrows_now.append(
                {
                    "src": atk_pos,
                    "dst": vic_pos,
                    "team": atk_team,
                    "color": (0.5, 0.5, 0.5),
                    "style": "solid",
                    "alpha": 0.9,
                    "text": "MISS",
                }
            )
            headline = line
            draw_board(arrows_now, headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.15)
            continue

        m = kill_re.match(line)
        if m:
            pos = int(m.group(2))
            if pos in state:
                state[pos]["hp"] = 0
            headline = line
            draw_board([], headline, active_pos=current_actor_pos)
            time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.1)
            continue

        if mage_banner_re.match(line) or poison_roll_re.match(line):
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

        draw_board([], line, active_pos=current_actor_pos)
        time.sleep((FRAME_DELAY * VISUAL_SPEED_MULT) / 1.2)

    draw_board([], "Конец эпизода (чекпоинт)", active_pos=None)