"""Находит seed эпизода, где отряд героя доходит до Адского рыцаря (тир 4), в окружении,
идентичном визуализатору, и запускает графическую визуализацию этого seed."""
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from campaign_env import CampaignEnv
from grid import DEFAULT_GRID_SIZE
from train_campaign import REWARD_CONFIG
from visualize_campaign import run_campaign_visualization

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "offline-1782077003"
MAX_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 80
TARGET = "Адский рыцарь"
MAX_GRID_STEPS = 2500

MODEL = f"./campaign_models/{RUN_ID}/final_model.zip"
VEC = f"./campaign_models/{RUN_ID}/vecnormalize.pkl"


def mask_fn(env):
    return env.compute_action_mask()


def make_base():
    return CampaignEnv(
        grid_size=DEFAULT_GRID_SIZE, **REWARD_CONFIG, persist_blue_hp=True,
        log_enabled=False, detailed_step_info=False, freeze_dynamic_action_layout=True,
        scripted_capital_bot_enabled=False, empire_territory_enabled=True,
        campaign_objective="blue_dragon", max_grid_steps=MAX_GRID_STEPS, Realcapital=2,
    )


def main():
    norm = VecNormalize.load(VEC, DummyVecEnv([lambda: Monitor(ActionMasker(make_base(), mask_fn))]))
    norm.training = False
    norm.norm_reward = False
    model = MaskablePPO.load(MODEL)

    found = None
    for seed in range(MAX_SEEDS):
        # Свежее окружение на каждый seed — как в визуализаторе (без утечки состояния).
        base = make_base()
        env = ActionMasker(base, mask_fn)
        np.random.seed(seed)
        torch.manual_seed(seed)
        obs, _ = env.reset(seed=seed)
        done = False
        names = set()
        while not done:
            mask = base.compute_action_mask()
            obs_n = norm.normalize_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1))
            action, _ = model.predict(obs_n, deterministic=False,
                                      action_masks=np.asarray(mask, dtype=bool).reshape(1, -1))
            obs, _, term, trunc, _ = env.step(int(np.asarray(action).reshape(-1)[0]))
            done = term or trunc
            for u in base._get_blue_state():
                n = str(u.get("name", "")).strip()
                if n:
                    names.add(n)
        has_t4 = TARGET in names
        print(f"seed {seed}: {'<<< АДСКИЙ РЫЦАРЬ' if has_t4 else 'нет тир-4'}", flush=True)
        if has_t4:
            found = seed
            break

    if found is None:
        print(f"За {MAX_SEEDS} seed'ов тир-4 не найден.")
        return

    print(f"\n=== Найден seed {found} с Адским рыцарем — запускаю визуализацию ===", flush=True)
    run_campaign_visualization(
        model_path=MODEL,
        deterministic=False,
        campaign_objective="blue_dragon",
        scripted_capital_bot_enabled=False,
        vecnormalize_path=VEC,
        seed=found,
        max_grid_steps=MAX_GRID_STEPS,
        delay=0.15,
    )


if __name__ == "__main__":
    main()
