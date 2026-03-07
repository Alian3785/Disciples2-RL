import argparse
import os
from collections import Counter

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
from console_encoding import setup_utf8_console

setup_utf8_console()

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from campaign_env import CampaignEnv


def mask_fn(env: CampaignEnv) -> np.ndarray:
    return env.compute_action_mask()


def make_env() -> Monitor:
    base = CampaignEnv(
        grid_size=40,
        reward_engage_battle=0.1,
        reward_defeat_enemy=4.0,
        reward_all_enemies=4.0,
        reward_loss=-3.0,
        reward_timeout=-3.0,
        persist_blue_hp=True,
        log_enabled=False,
        max_grid_steps=1500,
        realcapital=2,
    )
    return Monitor(ActionMasker(base, mask_fn))


def evaluate_model(run_id: str, episodes: int, deterministic: bool, model_type: str) -> int:
    if model_type == "final":
        model_path = f"./campaign_models/{run_id}/final_model.zip"
        vec_paths = [
            f"./campaign_models/{run_id}/vecnormalize.pkl",
            f"./campaign_models/{run_id}/best/best_vecnormalize.pkl",
        ]
    else:
        model_path = f"./campaign_models/{run_id}/best/best_model.zip"
        vec_paths = [
            f"./campaign_models/{run_id}/best/best_vecnormalize.pkl",
            f"./campaign_models/{run_id}/vecnormalize.pkl",
        ]

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return 1

    env = DummyVecEnv([make_env])

    vec_path = next((path for path in vec_paths if os.path.exists(path)), None)
    if vec_path is not None:
        env = VecNormalize.load(vec_path, env)
    else:
        print("[WARN] VecNormalize stats not found, running without them.")

    env.training = False
    env.norm_reward = False

    model = MaskablePPO.load(model_path, env=env)

    episode_rewards = []
    episode_enemies_defeated = []
    outcomes = Counter()

    for _ in range(int(episodes)):
        obs = env.reset()
        done = np.array([False])
        total_reward = 0.0
        last_info = {}
        defeated_enemy_ids = set()

        while not done[0]:
            action_masks = get_action_masks(env)
            action, _ = model.predict(
                obs, deterministic=deterministic, action_masks=action_masks
            )
            obs, reward, done, infos = env.step(action)
            total_reward += float(reward[0])
            last_info = infos[0]
            if last_info.get("battle_result") == "victory":
                enemy_id = last_info.get("enemy_id")
                if enemy_id is not None:
                    defeated_enemy_ids.add(int(enemy_id))

        episode_rewards.append(total_reward)
        episode_enemies_defeated.append(len(defeated_enemy_ids))
        outcomes[last_info.get("campaign_result", "other")] += 1

    print(f"run_id: {run_id}")
    print(f"model_type: {model_type}")
    print(f"episodes: {episodes}")
    print(
        f"mean_reward: {np.mean(episode_rewards):.3f} +/- {np.std(episode_rewards):.3f}"
    )
    print(
        f"mean_enemies_defeated: {np.mean(episode_enemies_defeated):.3f} +/- {np.std(episode_enemies_defeated):.3f}"
    )
    print(f"wins: {outcomes.get('victory', 0)}")
    print(f"defeats: {outcomes.get('defeat', 0)}")
    print(f"timeouts: {outcomes.get('timeout', 0)}")
    print(f"other: {outcomes.get('other', 0)}")
    print("episode_rewards:", [round(x, 3) for x in episode_rewards])
    print("enemies_defeated_per_episode:", episode_enemies_defeated)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quick eval of campaign model: mean reward + victory/defeat/timeout counts."
    )
    parser.add_argument("--run-id", type=str, default="offline-1770462365")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["best", "final"],
        default="best",
        help="Which checkpoint to evaluate",
    )
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    return evaluate_model(
        run_id=args.run_id,
        episodes=args.episodes,
        deterministic=args.deterministic,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    raise SystemExit(main())

