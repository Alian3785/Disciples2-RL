"""Десяток СТОХАСТИЧЕСКИХ эпизодов последней обученной модели (blue_dragon, 10M).

По каждому эпизоду печатает:
  - сколько отрядов врагов побеждено (defeated/total на карте),
  - до какого тира прокачаны юниты героя (пиковый тир + финальный ростер по тирам),
  - побеждён ли синий дракон (campaign_result == "victory"); плюс «дошёл до дракона».

Модель/нормировка по умолчанию — финальная 10M последнего прогона.
  python tools/eval_episodes_stochastic.py [episodes] [model.zip] [vecnormalize.pkl]
"""
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from campaign_env import CampaignEnv
from grid import DEFAULT_GRID_SIZE

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MODEL_PATH = sys.argv[2] if len(sys.argv) > 2 else "campaign_models/offline-1782142870/final_model.zip"
VECNORM_PATH = sys.argv[3] if len(sys.argv) > 3 else "campaign_models/offline-1782142870/vecnormalize.pkl"
# 4-й аргумент: "det"/"deterministic"/"1" -> детерминированная политика; иначе стохастическая.
DETERMINISTIC = (
    str(sys.argv[4]).lower() in ("det", "deterministic", "1", "true", "yes")
    if len(sys.argv) > 4 else False
)
STEP_CAP = 5000


def mask_fn(env):
    return env.compute_action_mask()


def make_env():
    base = CampaignEnv(
        grid_size=DEFAULT_GRID_SIZE, persist_blue_hp=True, log_enabled=False,
        freeze_dynamic_action_layout=True, empire_territory_enabled=True,
        scripted_capital_bot_enabled=False, campaign_objective="blue_dragon",
        max_grid_steps=3000, Realcapital=2,
    )
    return Monitor(ActionMasker(base, mask_fn))


def roster(base):
    return [
        (str(u.get("name", "")), int(u.get("Level", 0) or 0))
        for u in base._get_blue_state()
        if str(u.get("name", "")).strip() and str(u.get("name", "")).strip() != "пусто"
    ]


def enemies_defeated(base):
    ea = getattr(base.grid_env, "enemies_alive", {}) or {}
    total = len(ea)
    defeated = sum(1 for alive in ea.values() if not bool(alive))
    return defeated, total


def main():
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(VECNORM_PATH, env)
    env.training = False
    env.norm_reward = False
    base = env.venv.envs[0].unwrapped
    model = MaskablePPO.load(MODEL_PATH, env=env)

    print(f"model: {MODEL_PATH}")
    print(f"vecnorm: {VECNORM_PATH}")
    print(f"objective: blue_dragon | episodes: {EPISODES} | "
          f"mode: {'DETERMINISTIC' if DETERMINISTIC else 'STOCHASTIC'}\n")

    wins = 0
    for ep in range(1, EPISODES + 1):
        obs = env.reset()
        done = np.array([False])
        name_max_level, steps, got_dragon, result = {}, 0, False, ""
        # Снимки состояния ДО шага (до авто-сброса VecEnv на терминальном шаге).
        prev_roster, prev_def = roster(base), enemies_defeated(base)
        term_info = None
        while not done[0] and steps < STEP_CAP:
            snap_roster = roster(base)
            prev_roster, prev_def = snap_roster, enemies_defeated(base)
            for name, lvl in snap_roster:
                if lvl > name_max_level.get(name, -1):
                    name_max_level[name] = lvl
            masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=DETERMINISTIC, action_masks=masks)
            obs, _, done, infos = env.step(action)
            steps += 1
            info = infos[0]
            if info.get("green_dragon_reached"):
                got_dragon = True
            if done[0]:
                result = str(info.get("campaign_result", "") or "")
                term_info = info

        if term_info is None:
            # Эпизод упёрся в лимит шагов: авто-сброса не было, читаем напрямую.
            defeated, total = enemies_defeated(base)
            final_roster = roster(base)
        elif "enemies_defeated_count" in term_info:
            # Терминальный info содержит точный счёт побеждённых (до сброса).
            defeated = int(term_info.get("enemies_defeated_count", 0))
            total = int(term_info.get("enemies_defeated_total", 0))
            final_roster = prev_roster
        else:
            defeated, total = prev_def
            final_roster = prev_roster
        peak_tier = max(name_max_level.values()) if name_max_level else 0
        dragon_killed = result == "victory"
        if dragon_killed:
            wins += 1
        tier_counts = Counter(l for _, l in final_roster)
        tiers_str = ", ".join(f"тир{t}×{tier_counts[t]}" for t in sorted(tier_counts, reverse=True))
        dragon_str = "ДА" if dragon_killed else ("дошёл, не убил" if got_dragon else "нет")
        units_str = ", ".join(f"{n}(т{l})" for n, l in sorted(final_roster, key=lambda x: -x[1]))
        print(f"Эп {ep:2d}: побеждено отрядов {defeated}/{total} | "
              f"пик тир {peak_tier} | дракон: {dragon_str} | {steps} шаг")
        print(f"        финальный ростер [{tiers_str}]: {units_str}")

    print(f"\nИтого: дракон побеждён в {wins}/{EPISODES} эпизодах "
          f"({100.0 * wins / EPISODES:.0f}%)")


if __name__ == "__main__":
    main()
