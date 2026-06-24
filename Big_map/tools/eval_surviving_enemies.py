"""Прогоняет модель (blue_dragon) и собирает, какие enemy_id остаются ЖИВЫМИ к концу эпизода,
с их описанием и суммарным HP отряда — чтобы понять, стена это по силе или обрыв по шагам."""
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
from enemy_configs import ENEMY_CONFIGS, ENEMY_DESCRIPTIONS

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "offline-1782073485"
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 10


def mask_fn(env):
    return env.compute_action_mask()


def make_env():
    base = CampaignEnv(
        grid_size=DEFAULT_GRID_SIZE, persist_blue_hp=True, log_enabled=False,
        freeze_dynamic_action_layout=True, empire_territory_enabled=True,
        scripted_capital_bot_enabled=False, campaign_objective="blue_dragon",
        max_grid_steps=2500, Realcapital=2,
    )
    return Monitor(ActionMasker(base, mask_fn))


def enemy_hp(eid):
    team = ENEMY_CONFIGS.get(int(eid), [])
    return int(sum(float(u.get("max_health", 0) or 0) for u in team))


def main():
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(f"./campaign_models/{RUN_ID}/vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False
    base = env.venv.envs[0].unwrapped
    model = MaskablePPO.load(f"./campaign_models/{RUN_ID}/final_model.zip", env=env)

    survived = Counter()
    alive_end, defeated_end, outcomes = [], [], Counter()
    for _ in range(EPISODES):
        obs = env.reset()
        done = np.array([False])
        info = {}
        last_alive = dict(base.grid_env.enemies_alive)  # снимок ДО авто-ресета
        while not done[0]:
            last_alive = dict(base.grid_env.enemies_alive)  # состояние перед шагом
            masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, _, done, infos = env.step(action)
            info = infos[0]
            if isinstance(info.get("enemies_alive"), dict):
                last_alive = dict(info["enemies_alive"])  # точное состояние из терминального info
        ea = last_alive
        alive_ids = [int(k) for k, v in ea.items() if v]
        for k in alive_ids:
            survived[k] += 1
        alive_end.append(len(alive_ids))
        defeated_end.append(sum(1 for v in ea.values() if not v))
        outcomes[info.get("campaign_result", "other")] += 1
        del ea

    print(f"run_id: {RUN_ID} | episodes: {EPISODES} | objective: blue_dragon")
    print(f"в среднем побеждено: {np.mean(defeated_end):.1f} | осталось живых: {np.mean(alive_end):.1f} "
          f"(всего врагов {len(base.grid_env.enemies_alive)})")
    print(f"исходы: {dict(outcomes)}")
    print("\nЧаще всего ВЫЖИВАЮТ к концу эпизода (enemy_id: в N/эп эпизодах | HP отряда | описание):")
    for eid, cnt in survived.most_common(25):
        desc = ENEMY_DESCRIPTIONS.get(eid, "?")
        print(f"  enemy_{eid}: {cnt}/{EPISODES} | HP={enemy_hp(eid)} | {desc}")


if __name__ == "__main__":
    main()
