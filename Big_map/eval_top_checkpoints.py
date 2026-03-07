import argparse
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
from console_encoding import setup_utf8_console

setup_utf8_console()

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.ppo_mask import MaskablePPO

from bestmodeltest import make_env


def _extract_step(model_path: Path) -> int:
    stem = model_path.stem
    prefix = "campaign_ppo_step_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected checkpoint name: {model_path.name}")
    return int(stem[len(prefix) :])


def _resolve_run_dir(base_dir: Path, run_dir_arg: str | None) -> Path:
    checkpoints_root = base_dir / "campaign_checkpoints"
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
        if not run_dir.is_absolute():
            run_dir = checkpoints_root / run_dir
    else:
        run_dirs = [path for path in checkpoints_root.iterdir() if path.is_dir()]
        if not run_dirs:
            raise FileNotFoundError(f"No run directories found in {checkpoints_root}")
        run_dir = max(run_dirs, key=lambda path: path.stat().st_mtime)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def _evaluate_checkpoint(
    model_path: Path,
    vec_path: Path,
    episodes: int,
    deterministic: bool,
) -> dict:
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(str(vec_path), env)
    env.training = False
    env.norm_reward = False

    model = MaskablePPO.load(str(model_path), env=env)

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
                obs,
                deterministic=deterministic,
                action_masks=action_masks,
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

    env.close()

    return {
        "model_path": str(model_path),
        "vec_path": str(vec_path),
        "wins": outcomes.get("victory", 0),
        "defeats": outcomes.get("defeat", 0),
        "timeouts": outcomes.get("timeout", 0),
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_enemies": float(np.mean(episode_enemies_defeated)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the top-N latest campaign checkpoints and pick the best one."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Specific run directory inside campaign_checkpoints. Defaults to the latest one.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many latest checkpoints to evaluate.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Evaluation episodes per checkpoint.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic policy rollout.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    run_dir = _resolve_run_dir(base_dir, args.run_dir)

    candidates = sorted(
        run_dir.glob("campaign_ppo_step_*.zip"),
        key=_extract_step,
        reverse=True,
    )[: max(1, int(args.top_k))]

    if not candidates:
        raise FileNotFoundError(f"No checkpoint models found in {run_dir}")

    results = []
    for model_path in candidates:
        vec_path = model_path.with_name(
            model_path.name.replace("campaign_ppo_step_", "vecnormalize_step_").replace(".zip", ".pkl")
        )
        if not vec_path.exists():
            print(f"skip: missing vecnormalize for {model_path.name}")
            continue

        row = _evaluate_checkpoint(
            model_path=model_path,
            vec_path=vec_path,
            episodes=args.episodes,
            deterministic=not args.stochastic,
        )
        results.append(row)
        print(
            f"{model_path.name}: wins={row['wins']}, "
            f"mean_enemies={row['mean_enemies']:.3f}, mean_reward={row['mean_reward']:.3f}"
        )

    if not results:
        raise FileNotFoundError("No valid checkpoints with matching VecNormalize stats found.")

    results.sort(
        key=lambda row: (row["wins"], row["mean_enemies"], row["mean_reward"]),
        reverse=True,
    )

    best = results[0]
    print(f"\nrun_dir: {run_dir}")
    print(f"episodes_per_model: {args.episodes}")
    print("\nSummary:")
    for row in results:
        print(
            f"{Path(row['model_path']).name}: wins={row['wins']}, defeats={row['defeats']}, "
            f"timeouts={row['timeouts']}, mean_enemies={row['mean_enemies']:.3f}, "
            f"mean_reward={row['mean_reward']:.3f}"
        )

    print("\nBEST MODEL:")
    print(f"model_path: {best['model_path']}")
    print(f"vec_path: {best['vec_path']}")
    print(f"wins: {best['wins']}")
    print(f"mean_enemies: {best['mean_enemies']:.3f}")
    print(f"mean_reward: {best['mean_reward']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
