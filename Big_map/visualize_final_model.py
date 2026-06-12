# visualize_final_model.py
"""
Запуск кампании с готовой моделью MaskablePPO.
Два режима:
- Текстовый (по умолчанию)
- Графический (matplotlib) через флаг --viz

Если --model-path не указан, подставляется последний по дате файла
best/best_model.zip или final_model.zip в ./campaign_models (см. --checkpoint).
"""

import argparse
import os
import sys
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


def find_latest_checkpoint(
    model_kind: str,
    base_dir: str = "campaign_models",
) -> str | None:
    """Последний по mtime чекпоинт: ``final`` (final_model.zip) или ``best`` (best/best_model.zip)."""
    if not os.path.isdir(base_dir):
        return None
    candidates: list[tuple[str, float]] = []
    for name in os.listdir(base_dir):
        run_dir = os.path.join(base_dir, name)
        if not os.path.isdir(run_dir):
            continue
        if model_kind == "best":
            path = os.path.join(run_dir, "best", "best_model.zip")
        else:
            path = os.path.join(run_dir, "final_model.zip")
        if os.path.isfile(path):
            candidates.append((path, os.path.getmtime(path)))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])[0]


def resolve_model_path(model_path: str | None, checkpoint: str) -> str:
    if model_path:
        return model_path
    path = find_latest_checkpoint(checkpoint)
    if path:
        print(f"[model] {checkpoint}: {path}", file=sys.stderr)
        return path
    if checkpoint == "best":
        fallback = find_latest_checkpoint("final")
        if fallback:
            print(
                "[model] best_model.zip не найден, используется последний final_model.zip: "
                f"{fallback}",
                file=sys.stderr,
            )
            return fallback
    raise SystemExit(
        f"Не найдена модель в {os.path.abspath('campaign_models')}. "
        "Укажите --model-path или обучите модель."
    )


def resolve_vecnormalize_path(model_path: str, explicit_path: str | None = None) -> str | None:
    if explicit_path:
        return explicit_path

    model_path_abs = os.path.abspath(model_path)
    model_dir = os.path.dirname(model_path_abs)
    run_dir = os.path.dirname(model_dir) if os.path.basename(model_dir) == "best" else model_dir
    candidates = [
        os.path.join(model_dir, "best_vecnormalize.pkl"),
        os.path.join(run_dir, "vecnormalize.pkl"),
        os.path.join(run_dir, "best", "best_vecnormalize.pkl"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def run_episode(
    model: MaskablePPO,
    env: Monitor,
    max_steps: int,
    deterministic: bool,
    sleep_s: float,
    *,
    quiet: bool = False,
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
        if not quiet:
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
        default=None,
        help="Путь к .zip модели MaskablePPO (если не задан — авто из campaign_models)",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("best", "final"),
        default="best",
        help="Какой чекпоинт брать при авто-поиске: best_model или final_model",
    )
    parser.add_argument("--steps", type=int, default=500, help="Максимум шагов за эпизод")
    parser.add_argument("--episodes", type=int, default=1, help="Сколько эпизодов запустить подряд")
    parser.add_argument(
        "--quiet-eval",
        action="store_true",
        help="Без логов кампании и без env.render(); по эпизоду — компактная строка ПОБЕДА/НЕТ",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Использовать детерминированные действия (по умолчанию стохастические)",
    )
    parser.add_argument(
        "--no-scripted-bot",
        action="store_true",
        help="Отключить scripted capital bot в визуализации",
    )
    parser.add_argument(
        "--dragon",
        action="store_true",
        help="Использовать green-dragon objective, как в dragon training runs",
    )
    parser.add_argument(
        "--vecnormalize-path",
        default=None,
        help="Путь к VecNormalize .pkl; если не задан, ищется рядом с моделью",
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

    model_file = resolve_model_path(args.model_path, args.checkpoint)
    vecnormalize_file = resolve_vecnormalize_path(model_file, args.vecnormalize_path)
    campaign_objective = "dragon" if args.dragon else "cities"

    # Графическая визуализация (matplotlib)
    if args.viz:
        from visualize_campaign import run_campaign_visualization

        run_campaign_visualization(
            model_path=model_file,
            delay=args.sleep or 0.3,
            deterministic=args.deterministic,
            map_png_path=args.map_png,
            scenario_path=args.scenario_path,
            scripted_capital_bot_enabled=not args.no_scripted_bot,
            campaign_objective=campaign_objective,
            vecnormalize_path=vecnormalize_file,
        )
        return

    log_ok = not args.quiet_eval
    env = Monitor(
        ActionMasker(
            CampaignEnv(
                log_enabled=log_ok,
                Realcapital=2,
                scripted_capital_bot_enabled=not args.no_scripted_bot,
                campaign_objective=campaign_objective,
            ),
            mask_fn,
        ),
    )
    model = MaskablePPO.load(model_file, env=env)

    wins = 0
    for ep in range(1, args.episodes + 1):
        if not args.quiet_eval:
            print(f"\n=== Эпизод {ep}/{args.episodes} ===")
        steps_done, info = run_episode(
            model=model,
            env=env,
            max_steps=args.steps,
            deterministic=args.deterministic,
            sleep_s=args.sleep,
            quiet=args.quiet_eval,
        )
        campaign_result = info.get("campaign_result")
        is_win = campaign_result == "victory"
        wins += int(is_win)
        if args.quiet_eval:
            outcome = "ПОБЕДА" if is_win else "НЕТ"
            print(
                f"Эпизод {ep}/{args.episodes}: {outcome}  "
                f"(campaign_result={campaign_result}, шагов={steps_done})"
            )
        else:
            print(f"Эпизод завершён за {steps_done} шаг(ов). Info: {info}")

    if args.quiet_eval and args.episodes > 0:
        print(
            f"Итого побед в кампании: {wins}/{args.episodes} "
            f"({100.0 * wins / args.episodes:.1f}%)",
        )

    env.close()


if __name__ == "__main__":
    main()

