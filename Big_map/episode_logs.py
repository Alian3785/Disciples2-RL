from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from campaign_env import CampaignEnv
from console_encoding import setup_utf8_console


setup_utf8_console()


def mask_fn(env: CampaignEnv) -> np.ndarray:
    return env.compute_action_mask()


def find_latest_best_model(base_dir: Path) -> Path:
    candidates = sorted(
        base_dir.glob("campaign_models/*/best/best_model.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find any best model under {base_dir / 'campaign_models'}"
        )
    return candidates[0]


def unwrap_campaign_env(env: Monitor) -> CampaignEnv:
    current = env
    visited = set()
    while hasattr(current, "env") and id(current) not in visited:
        visited.add(id(current))
        current = current.env
    if not isinstance(current, CampaignEnv):
        raise TypeError(f"Expected CampaignEnv, got {type(current)!r}")
    return current


def drain_logs(base_env: CampaignEnv, *, step: Optional[int] = None) -> None:
    prefix = "" if step is None else f"[STEP {step}] "
    for line in base_env.pop_campaign_logs():
        print(f"{prefix}{line}")
    for line in base_env.pop_battle_logs():
        print(f"{prefix}{line}")


def run_episode(
    model: MaskablePPO,
    env: Monitor,
    *,
    max_steps: int,
    deterministic: bool,
    sleep_s: float,
    show_initial_logs: bool,
) -> Tuple[int, dict]:
    base_env = unwrap_campaign_env(env)
    obs, _ = env.reset()
    if show_initial_logs:
        drain_logs(base_env)
    else:
        base_env.pop_campaign_logs()
        base_env.pop_battle_logs()

    for step in range(1, max_steps + 1):
        action_masks = get_action_masks(env)
        action, _ = model.predict(
            obs,
            deterministic=deterministic,
            action_masks=action_masks,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        drain_logs(base_env, step=step)
        if sleep_s > 0:
            time.sleep(sleep_s)
        if terminated or truncated:
            final_info = dict(info)
            final_info["episode_reward_last_step"] = float(reward)
            return step, final_info

    return max_steps, {"campaign_result": "step_limit_reached"}


def build_env(realcapital: int) -> Monitor:
    base = CampaignEnv(log_enabled=False, realcapital=realcapital)
    return Monitor(ActionMasker(base, mask_fn))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run campaign model episodes and print event logs without map rendering."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a MaskablePPO .zip model. If omitted, the latest best_model.zip is used.",
    )
    parser.add_argument("--steps", type=int, default=5000, help="Maximum steps per episode.")
    parser.add_argument("--episodes", type=int, default=1, help="How many episodes to run.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions instead of stochastic sampling.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional pause between steps in seconds.",
    )
    parser.add_argument(
        "--realcapital",
        type=int,
        default=2,
        help="Faction id: 1=empire, 2=legions, 3=mountain_clans, 4=undead_hordes, 5=elves.",
    )
    parser.add_argument(
        "--show-initial-logs",
        action="store_true",
        help="Print initial reset logs with map setup details before the first step.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    model_path = Path(args.model_path).resolve() if args.model_path else find_latest_best_model(base_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    env = build_env(realcapital=int(args.realcapital))
    model = MaskablePPO.load(str(model_path), env=env)

    print(f"Model: {model_path}")
    print(f"Realcapital: {int(args.realcapital)}")
    print(f"Deterministic: {bool(args.deterministic)}")
    print(f"Episodes: {int(args.episodes)}")
    print(f"Step limit: {int(args.steps)}")

    for episode_index in range(1, int(args.episodes) + 1):
        print(f"\n=== Episode {episode_index}/{int(args.episodes)} ===")
        steps_done, info = run_episode(
            model=model,
            env=env,
            max_steps=int(args.steps),
            deterministic=bool(args.deterministic),
            sleep_s=float(args.sleep),
            show_initial_logs=bool(args.show_initial_logs),
        )
        print(f"[EPISODE END] steps={steps_done}")
        print(info)

    env.close()


if __name__ == "__main__":
    main()
