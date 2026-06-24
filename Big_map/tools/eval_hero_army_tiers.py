"""За N эпизодов финальной модели (blue_dragon) фиксирует пиковый состав отряда героя:
максимальный достигнутый тир (Level) и имена юнитов на этом тире — по каждому эпизоду.
"""
import os
import sys
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

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "offline-1782040607"
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 20
# 3-й аргумент: "stoch"/"0"/"false" -> стохастический выбор действий; иначе детерминированный.
DETERMINISTIC = (
    str(sys.argv[3]).lower() not in ("stoch", "0", "false", "no")
    if len(sys.argv) > 3
    else True
)


def mask_fn(env):
    return env.compute_action_mask()


def make_env():
    base = CampaignEnv(
        grid_size=DEFAULT_GRID_SIZE, persist_blue_hp=True, log_enabled=False,
        freeze_dynamic_action_layout=True, empire_territory_enabled=True,
        scripted_capital_bot_enabled=False, campaign_objective="blue_dragon",
        max_grid_steps=1800, Realcapital=2,
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
    env = VecNormalize.load(f"./campaign_models/{RUN_ID}/vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False
    base = env.venv.envs[0].unwrapped
    model = MaskablePPO.load(f"./campaign_models/{RUN_ID}/final_model.zip", env=env)

    peak_tiers = []
    print(f"run_id: {RUN_ID} | model: final | objective: blue_dragon | episodes: {EPISODES}\n")
    for ep in range(1, EPISODES + 1):
        obs = env.reset()
        done = np.array([False])
        name_max_level = {}  # имя -> макс. тир за эпизод
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

    print(f"\nСредний пиковый тир за эпизод: {np.mean(peak_tiers):.2f}")
    from collections import Counter
    dist = Counter(peak_tiers)
    print("Распределение пикового тира:", {k: dist[k] for k in sorted(dist)})


if __name__ == "__main__":
    main()
