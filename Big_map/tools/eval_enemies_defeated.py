"""Eval финальной модели в окружении blue_dragon: среднее число побеждённых врагов за эпизод.

Окружение строится с теми же флагами, что и при обучении (freeze layout, empire territory,
Realcapital=2, max_grid_steps=1800, без скриптового бота), чтобы obs/action совпадали.
Считаются уникальные enemy_id, побеждённые за эпизод (как в bestmodeltest.py).
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

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "offline-1782040607"
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 12
# Опционально явные пути к модели и vecnormalize (3-й и 4-й аргументы).
MODEL_PATH = sys.argv[3] if len(sys.argv) > 3 else None
VEC_PATH = sys.argv[4] if len(sys.argv) > 4 else None


def mask_fn(env):
    return env.compute_action_mask()


def make_env():
    base = CampaignEnv(
        grid_size=DEFAULT_GRID_SIZE,
        persist_blue_hp=True,
        log_enabled=False,
        freeze_dynamic_action_layout=True,
        empire_territory_enabled=True,
        scripted_capital_bot_enabled=False,
        campaign_objective="blue_dragon",
        max_grid_steps=1800,
        Realcapital=2,
    )
    return Monitor(ActionMasker(base, mask_fn))


def main():
    model_path = MODEL_PATH or f"./campaign_models/{RUN_ID}/final_model.zip"
    vec_path = VEC_PATH or f"./campaign_models/{RUN_ID}/vecnormalize.pkl"
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vec_path, env)
    env.training = False
    env.norm_reward = False
    model = MaskablePPO.load(model_path, env=env)

    enemies_defeated, dragon_hits, outcomes = [], [], Counter()
    for _ in range(EPISODES):
        obs = env.reset()
        done = np.array([False])
        defeated = set()
        last_info = {}
        while not done[0]:
            masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, _, done, infos = env.step(action)
            last_info = infos[0]
            if last_info.get("battle_result") == "victory":
                eid = last_info.get("enemy_id")
                if eid is not None:
                    defeated.add(int(eid))
        enemies_defeated.append(len(defeated))
        dragon_hits.append(1 if 31 in defeated else 0)
        outcomes[last_info.get("campaign_result", "other")] += 1

    print(f"run_id: {RUN_ID} | model: final | episodes: {EPISODES} | objective: blue_dragon")
    print(f"mean_enemies_defeated: {np.mean(enemies_defeated):.2f} +/- {np.std(enemies_defeated):.2f}")
    print(f"per_episode: {enemies_defeated}")
    print(f"dragon_killed_episodes: {sum(dragon_hits)}/{EPISODES}")
    print(f"outcomes: {dict(outcomes)}")


if __name__ == "__main__":
    main()
