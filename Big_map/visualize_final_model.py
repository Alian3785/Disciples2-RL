# visualize_final_model.py
"""
Запуск кампании с готовой моделью MaskablePPO.
Два режима:
- Текстовый (по умолчанию)
- Графический (matplotlib) через флаг --viz

По умолчанию берёт финальный чекпоинт:
./campaign_models/offline-1765058113/final_model.zip
"""

import argparse
import time
from typing import Tuple

import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from campaign_env import CampaignEnv
from console_encoding import setup_utf8_console


setup_utf8_console()


def mask_fn(env: CampaignEnv) -> np.ndarray:
    """Маска допустимых действий для ActionMasker."""
    return env.compute_action_mask()


def run_episode(
    model: MaskablePPO,
    env: Monitor,
    max_steps: int,
    deterministic: bool,
    sleep_s: float,
) -> Tuple[int, dict]:
    """Выполняет один эпизод и печатает состояние после каждого шага."""
    obs, _ = env.reset()

    for step in range(max_steps):
        action_masks = get_action_masks(env)
        action, _ = model.predict(
            obs,
            deterministic=deterministic,
            action_masks=action_masks,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if sleep_s > 0:
            time.sleep(sleep_s)
        if terminated or truncated:
            return step + 1, info

    return max_steps, {"campaign_result": "truncated"}


def main():
    parser = argparse.ArgumentParser(description="Визуализация финальной модели Campaign")
    parser.add_argument(
        "--model-path",
        default="./campaign_models/offline-1765058113/final_model.zip",
        help="Путь к .zip модели MaskablePPO",
    )
    parser.add_argument("--steps", type=int, default=500, help="Максимум шагов за эпизод")
    parser.add_argument("--episodes", type=int, default=1, help="Сколько эпизодов запустить подряд")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Использовать детерминированные действия (по умолчанию стохастические)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Пауза (сек) между шагами для замедления вывода",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Включить графическую визуализацию (matplotlib из visualize_campaign.py)",
    )
    parser.add_argument(
        "--map-png",
        default="./outputs/a_return_to_simpler_times_entries.png",
        help="Путь к PNG-фону карты для режима --viz",
    )
    parser.add_argument(
        "--scenario-path",
        default=(
            "D:/Disciples II "
            "\u0412\u043e\u0441\u0441\u0442\u0430\u043d\u0438\u0435 "
            "\u042d\u043b\u044c\u0444\u043e\u0432/Exports/A Return To Simpler Times.sg"
        ),
        help="Путь к .sg сценарию для подсветки столиц и городов в режиме --viz",
    )
    args = parser.parse_args()

    # Графическая визуализация (matplotlib)
    if args.viz:
        from visualize_campaign import run_campaign_visualization

        run_campaign_visualization(
            model_path=args.model_path,
            delay=args.sleep or 0.3,
            deterministic=args.deterministic,
            map_png_path=args.map_png,
            scenario_path=args.scenario_path,
        )
        return

    env = Monitor(ActionMasker(CampaignEnv(log_enabled=True, realcapital=2), mask_fn))
    model = MaskablePPO.load(args.model_path, env=env)

    for ep in range(1, args.episodes + 1):
        print(f"\n=== Эпизод {ep}/{args.episodes} ===")
        steps_done, info = run_episode(
            model=model,
            env=env,
            max_steps=args.steps,
            deterministic=args.deterministic,
            sleep_s=args.sleep,
        )
        print(f"Эпизод завершён за {steps_done} шаг(ов). Info: {info}")

    env.close()


if __name__ == "__main__":
    main()

