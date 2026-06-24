"""Сравнение двух чекпоинтов blue_dragon бок о бок: база 5.0M (старт дообучения) и 10.0M (после +5M).

За N эпизодов на каждой модели (детерминированно, одинаковая среда как в обучении)
считает: victory-rate над синим драконом, долю эпизодов с выходом на дракона,
распределение пикового тира отряда и среднюю длину эпизода.

Жёсткий предел 5000 env-шагов на эпизод (как MAX_EVAL_EPISODE_STEPS в обучении),
чтобы циклы перестановок не делали эпизод бесконечным.

  python tools/eval_compare_5m_10m.py [episodes]
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

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 60
STEP_CAP = 5000

MODELS = [
    ("5.0M (база, старт дообучения)",
     "campaign_checkpoints/offline-1782139809/campaign_ppo_step_5000000.zip",
     "campaign_checkpoints/offline-1782139809/vecnormalize_step_5000000.pkl"),
    ("10.0M (после +5M)",
     "campaign_models/offline-1782142870/final_model.zip",
     "campaign_models/offline-1782142870/vecnormalize.pkl"),
]


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


def eval_model(label, model_path, vecnorm_path, episodes):
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    base = env.venv.envs[0].unwrapped
    model = MaskablePPO.load(model_path, env=env)

    peak, lengths = [], []
    victories = reached = capped = 0
    print(f"\n>>> {label}  ({episodes} эп.)")
    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = np.array([False])
        name_max_level, steps, got_dragon, result = {}, 0, False, ""
        while not done[0] and steps < STEP_CAP:
            masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, _, done, infos = env.step(action)
            steps += 1
            info = infos[0]
            if info.get("green_dragon_reached"):
                got_dragon = True
            if done[0]:
                result = str(info.get("campaign_result", "") or "")
            for name, lvl in roster(base):
                if lvl > name_max_level.get(name, -1):
                    name_max_level[name] = lvl
        if not done[0]:
            capped += 1
        peak_tier = max(name_max_level.values()) if name_max_level else 0
        peak.append(peak_tier)
        lengths.append(steps)
        if result == "victory":
            victories += 1
        if got_dragon:
            reached += 1
        flag = "ПОБЕДА" if result == "victory" else ("дошёл" if got_dragon else result or "лимит")
        print(f"  Эп {ep:2d}: тир {peak_tier} | {steps:4d} шаг | {flag}")
    return dict(label=label, n=episodes, peak=peak, victories=victories,
                reached=reached, capped=capped, lengths=lengths)


def report(r):
    n, peak = r["n"], r["peak"]
    dist = Counter(peak)
    print(f"\n=== {r['label']} ===")
    print(f"  Победа над синим драконом: {r['victories']}/{n} ({100*r['victories']/n:.1f}%)")
    print(f"  Вышел на дракона:          {r['reached']}/{n} ({100*r['reached']/n:.1f}%)")
    print(f"  Достигли лимита шагов:     {r['capped']}/{n} ({100*r['capped']/n:.1f}%)")
    print(f"  Средняя длина эпизода:     {np.mean(r['lengths']):.0f} env-шагов")
    print(f"  Средний пиковый тир:       {np.mean(peak):.2f}")
    print("  Распределение пикового тира:")
    for k in sorted(dist):
        print(f"    тир {k}: {dist[k]:2d}/{n} ({100*dist[k]/n:.1f}%)")


def main():
    results = [eval_model(*m, EPISODES) for m in MODELS]
    for r in results:
        report(r)
    a, b = results
    print("\n=== Δ (10M − 5M), п.п. / тир ===")
    print(f"  Победы:             {100*b['victories']/b['n'] - 100*a['victories']/a['n']:+.1f} п.п.")
    print(f"  Выход на дракона:   {100*b['reached']/b['n'] - 100*a['reached']/a['n']:+.1f} п.п.")
    print(f"  Средний пик. тир:   {np.mean(b['peak']) - np.mean(a['peak']):+.2f}")


if __name__ == "__main__":
    main()
