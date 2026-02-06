# train_campaign.py
"""
Скрипт обучения агента в Campaign Mode.

Агент учится:
1. Перемещаться по карте 10×10
2. Находить и побеждать 4 разных врагов
3. Сохранять HP между боями для оптимизации стратегии
"""

import os
import time
from pathlib import Path
import numpy as np

# Отключаем torch._dynamo, чтобы избежать зависимости от sympy/gmpy при обучении.
# Это стабилизирует запуск на окружениях без полной установки SymPy.
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

try:
    import wandb
except ImportError:
    wandb = None

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from wandb.integration.sb3 import WandbCallback
    
except ImportError:
    WandbCallback = None

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

try:
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    _HAS_MASKABLE_EVAL = True
except ImportError:
    _HAS_MASKABLE_EVAL = False

from campaign_env import CampaignEnv

# Для Windows и симлинков W&B
if os.name == "nt":
    os.environ.setdefault("WANDB_DISABLE_SYMLINKS", "true")

# Директория этого файла (чтобы пути были стабильны независимо от cwd)
BASE_DIR = Path(__file__).resolve().parent


def mask_fn(env: CampaignEnv) -> np.ndarray:
    """Функция маскирования для ActionMasker."""
    return env.compute_action_mask()


def make_env(log_enabled: bool = False):
    """Фабрика создания среды."""
    base = CampaignEnv(
        grid_size=10,
        reward_engage_battle=0.1,  # Награда за участие в битве (нашёл врага)
        reward_defeat_enemy=0.3,   # Награда за победу в каждой битве
        reward_all_enemies=4.0,    # Финальная награда за победу над всеми врагами
        reward_loss=0.0,           # Штраф за поражение
        reward_timeout=-6.0,       # Штраф за таймаут (увеличен в 3 раза)
        persist_blue_hp=True,
        log_enabled=log_enabled,
        max_grid_steps=200,
        realcapital=2,
    )
    masked = ActionMasker(base, mask_fn)
    return Monitor(masked)


class CampaignCheckpointCallback(BaseCallback):
    """Сохраняет модель каждые N шагов."""

    def __init__(
        self, save_dir: str, milestone_steps: int = 500_000, verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.milestone_steps = int(milestone_steps)
        self.next_milestone = self.milestone_steps
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_milestone:
            path = os.path.join(self.save_dir, f"campaign_ppo_step_{self.next_milestone}.zip")
            self.model.save(path)
            vec_norm = self.model.get_vec_normalize_env()
            if vec_norm is not None:
                vec_path = os.path.join(
                    self.save_dir, f"vecnormalize_step_{self.next_milestone}.pkl"
                )
                vec_norm.save(vec_path)
            if self.verbose:
                print(f"[Checkpoint] Saved at {self.next_milestone} steps > {path}")
            if wandb and wandb.run:
                wandb.log({"checkpoint_saved_at": self.next_milestone})
            self.next_milestone += self.milestone_steps
        return True


class SaveVecNormalizeOnBestCallback(BaseCallback):
    """Сохраняет статистики VecNormalize при обновлении лучшей модели."""

    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        vec_norm = self.model.get_vec_normalize_env()
        if vec_norm is not None:
            target = os.path.join(self.save_path, "best_vecnormalize.pkl")
            vec_norm.save(target)
            if self.verbose > 0:
                print(f"[VecNormalize] Saved best stats > {target}")
        return True


class CampaignMetricsCallback(BaseCallback):
    """Логирует метрики кампании в W&B."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.victories = 0
        self.defeats = 0
        self.timeouts = 0

    def _on_step(self) -> bool:
        # Проверяем завершённые эпизоды
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

            if info.get("campaign_result") == "victory":
                self.victories += 1
            elif info.get("campaign_result") == "defeat":
                self.defeats += 1
            elif info.get("campaign_result") == "timeout":
                self.timeouts += 1

        # Логируем каждые 1000 шагов
        if self.num_timesteps % 1000 == 0 and wandb and wandb.run:
            total_episodes = self.victories + self.defeats + self.timeouts
            if total_episodes > 0:
                wandb.log({
                    "campaign/victory_rate": self.victories / total_episodes,
                    "campaign/defeat_rate": self.defeats / total_episodes,
                    "campaign/timeout_rate": self.timeouts / total_episodes,
                    "campaign/total_episodes": total_episodes,
                })

        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Campaign Agent")
    parser.add_argument("--total-steps", type=int, default=2_000_000, help="Total training steps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--checkpoint-freq", type=int, default=500_000, help="Checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Evaluation frequency")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name")
    args = parser.parse_args()

    # Проверка среды
    print("Проверка среды CampaignEnv...")
    test_env = CampaignEnv(log_enabled=False, realcapital=2)
    check_env(test_env, warn=True)
    print("Среда прошла проверку!")

    # -------------------- Параметры --------------------
    TOTAL_STEPS = args.total_steps
    N_ENVS = args.n_envs
    CHECKPOINT_FREQ = args.checkpoint_freq
    EVAL_FREQ = args.eval_freq
    VECNORM_NORM_OBS = False
    VECNORM_NORM_REWARD = True
    VECNORM_CLIP_REWARD = 10.0

    # -------------------- Инициализация W&B --------------------
    WANDB_AVAILABLE = wandb is not None and WandbCallback is not None and not args.no_wandb

    run_name = args.run_name or f"campaign-ppo-{TOTAL_STEPS // 1000}k"

    if WANDB_AVAILABLE:
        try:
            run = wandb.init(
                project="disciples-campaign",
                name=run_name,
                config={
                    "algo": "MaskablePPO",
                    "total_timesteps": TOTAL_STEPS,
                    "n_envs": N_ENVS,
                    "n_steps": 2048,
                    "batch_size": 256,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "learning_rate": 3e-4,
                    "clip_range": 0.2,
                    "ent_coef": 0.05,
                    "checkpoint_freq": CHECKPOINT_FREQ,
                    "eval_freq": EVAL_FREQ,
                    "grid_size": 10,
                    "num_enemies": 4,
                    "persist_blue_hp": True,
                    "reward_engage_battle": 0.1,
                    "reward_defeat_enemy": 0.3,
                    "reward_all_enemies": 4.0,
                    "reward_loss": 0.0,
                    "reward_timeout": -6.0,
                    "vecnormalize_norm_obs": VECNORM_NORM_OBS,
                    "vecnormalize_norm_reward": VECNORM_NORM_REWARD,
                    "vecnormalize_clip_reward": VECNORM_CLIP_REWARD,
                },
                sync_tensorboard=True,
                save_code=True,
            )
            run_id = run.id
        except Exception as exc:
            WANDB_AVAILABLE = False
            run_id = f"offline-{int(time.time())}"
            print(f"[WARN] W&B init failed, switching to offline mode: {exc}")
    else:
        run_id = f"offline-{int(time.time())}"
        print("[INFO] W&B disabled, running in offline mode")

    # Директории
    CHECKPOINT_DIR = str(BASE_DIR / "campaign_checkpoints" / run_id)
    MODEL_DIR = f"./campaign_models/{run_id}"
    EVAL_DIR = f"./campaign_eval/{run_id}"
    TB_LOG_DIR = f"./campaign_tb_logs/{run_id}"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # -------------------- Создание сред --------------------
    print(f"Создание {N_ENVS} параллельных сред...")
    vec_env = make_vec_env(make_env, n_envs=N_ENVS)
    # Normalize only rewards to stabilize PPO targets; observation normalization stays disabled.
    vec_env = VecNormalize(
        vec_env,
        training=True,
        norm_obs=VECNORM_NORM_OBS,
        norm_reward=VECNORM_NORM_REWARD,
        clip_reward=VECNORM_CLIP_REWARD,
        gamma=0.99,
    )

    # Оценочная среда
    eval_env = DummyVecEnv([lambda: make_env(log_enabled=False)])
    # Eval env receives running statistics from train env via EvalCallback sync.
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=VECNORM_NORM_OBS,
        norm_reward=False,
        clip_reward=VECNORM_CLIP_REWARD,
        gamma=0.99,
    )

    # -------------------- Модель --------------------
    print("Инициализация MaskablePPO...")
    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.05,  # Увеличенная энтропия для исследования
        vf_coef=0.5,
        tensorboard_log=TB_LOG_DIR,
    )

    # -------------------- Callbacks --------------------
    callbacks = []

    # Чекпоинты
    checkpoint_cb = CampaignCheckpointCallback(
        save_dir=CHECKPOINT_DIR,
        milestone_steps=CHECKPOINT_FREQ,
        verbose=1,
    )
    callbacks.append(checkpoint_cb)

    # Метрики кампании
    metrics_cb = CampaignMetricsCallback(verbose=0)
    callbacks.append(metrics_cb)

    # W&B callback (без сохранения модели — на Windows symlink требует прав админа)
    if WANDB_AVAILABLE and WandbCallback:
        wandb_cb = WandbCallback(
            gradient_save_freq=10000,
            verbose=2,
        )
        callbacks.append(wandb_cb)

    # Evaluation callback
    best_vecnorm_cb = SaveVecNormalizeOnBestCallback(
        save_path=f"{MODEL_DIR}/best",
        verbose=0,
    )
    if _HAS_MASKABLE_EVAL:
        eval_cb = MaskableEvalCallback(           
            eval_env,
            best_model_save_path=f"{MODEL_DIR}/best",
            log_path=EVAL_DIR,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
            callback_on_new_best=best_vecnorm_cb,
        )
    else:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=f"{MODEL_DIR}/best",
            log_path=EVAL_DIR,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
            callback_on_new_best=best_vecnorm_cb,
        )
    callbacks.append(eval_cb)

    # -------------------- Обучение --------------------
    print(f"\n{'='*60}")
    print("Запуск обучения Campaign Agent")
    print(f"Всего шагов: {TOTAL_STEPS:,}")
    print(f"Параллельных сред: {N_ENVS}")
    print(f"Checkpoint каждые: {CHECKPOINT_FREQ:,} шагов")
    print(f"Evaluation каждые: {EVAL_FREQ:,} шагов")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=CallbackList(callbacks),
    )

    # -------------------- Сохранение финальной модели --------------------
    final_model_path = f"{MODEL_DIR}/final_model"
    model.save(final_model_path)
    vec_norm = model.get_vec_normalize_env()
    if vec_norm is not None:
        vec_norm_path = f"{MODEL_DIR}/vecnormalize.pkl"
        vec_norm.save(vec_norm_path)
        print(f"VecNormalize stats saved: {vec_norm_path}")
    print(f"\nФинальная модель сохранена: {final_model_path}.zip")

    # Итоговая статистика
    print(f"\n{'='*60}")
    print("ИТОГИ ОБУЧЕНИЯ:")
    print(f"  Побед: {metrics_cb.victories}")
    print(f"  Поражений: {metrics_cb.defeats}")
    print(f"  Таймаутов: {metrics_cb.timeouts}")
    total = metrics_cb.victories + metrics_cb.defeats + metrics_cb.timeouts
    if total > 0:
        print(f"  Winrate: {100 * metrics_cb.victories / total:.1f}%")
    print(f"{'='*60}")

    if WANDB_AVAILABLE and wandb.run:
        wandb.finish()
