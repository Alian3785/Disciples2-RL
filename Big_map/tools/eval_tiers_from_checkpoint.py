"""Оценка пикового тира отряда героя по ПРОИЗВОЛЬНОМУ чекпоинту (а не только final_model).

За N эпизодов фиксирует максимальный достигнутый Level (тир) среди юнитов героя
и строит распределение пикового тира с процентами. Env-конфиг совпадает с обучением
blue_dragon: без скриптового бота, империя вкл., layout заморожен.

Использование:
  python tools/eval_tiers_from_checkpoint.py <model.zip> <vecnormalize.pkl> [episodes] [stoch]
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

MODEL_PATH = sys.argv[1]
VECNORM_PATH = sys.argv[2]
EPISODES = int(sys.argv[3]) if len(sys.argv) > 3 else 30
DETERMINISTIC = (
    str(sys.argv[4]).lower() not in ("stoch", "0", "false", "no")
    if len(sys.argv) > 4
    else True
)


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


def main():
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(VECNORM_PATH, env)
    env.training = False
    env.norm_reward = False
    base = env.venv.envs[0].unwrapped
    model = MaskablePPO.load(MODEL_PATH, env=env)

    peak_tiers = []
    mode = "deterministic" if DETERMINISTIC else "stochastic"
    print(f"model: {MODEL_PATH}")
    print(f"vecnorm: {VECNORM_PATH}")
    print(f"objective: blue_dragon | episodes: {EPISODES} | mode: {mode}\n")
    for ep in range(1, EPISODES + 1):
        obs = env.reset()
        done = np.array([False])
        name_max_level = {}
        while not done[0]:
            masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=DETERMINISTIC, action_masks=masks)
            obs, _, done, _ = env.step(action)
            for name, lvl in roster(base):
                if lvl > name_max_level.get(name, -1):
                    name_max_level[name] = lvl
        peak = max(name_max_level.values()) if name_max_level else 0
        peak_tiers.append(peak)
        top_units = sorted(n for n, l in name_max_level.items() if l == peak)
        print(f"Эп {ep:2d}: макс. тир {peak} — {', '.join(top_units)}")

    n = len(peak_tiers)
    print(f"\nСредний пиковый тир за эпизод: {np.mean(peak_tiers):.2f}")
    dist = Counter(peak_tiers)
    print("\nРаспределение пикового тира (доля эпизодов):")
    for k in sorted(dist):
        cnt = dist[k]
        print(f"  тир {k}: {cnt:2d}/{n}  ({100.0 * cnt / n:.1f}%)")
    print("\nДоля эпизодов, где достигнут тир >= X:")
    for k in sorted(dist):
        atleast = sum(c for t, c in dist.items() if t >= k)
        print(f"  >= тир {k}: {100.0 * atleast / n:.1f}%")


if __name__ == "__main__":
    main()
