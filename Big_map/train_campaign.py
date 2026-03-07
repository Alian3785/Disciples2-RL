# train_campaign.py
"""
Скрипт обучения агента в Campaign Mode.

Агент учится:
1. Перемещаться по карте 40×40
2. Находить и побеждать 4 разных врагов
3. Сохранять HP между боями для оптимизации стратегии
"""

import os
import re
import time
import subprocess
from pathlib import Path
import numpy as np
from console_encoding import setup_utf8_console

# Отключаем torch._dynamo, чтобы избежать зависимости от sympy/gmpy при обучении.
# Это стабилизирует запуск на окружениях без полной установки SymPy.
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
setup_utf8_console()

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
DEFAULT_GRID_SIZE = 40

REWARD_CONFIG = {
    "reward_engage_battle": 0.25,
    "reward_defeat_enemy": 4.0,
    "reward_all_enemies": 8.0,
    "reward_loss": -6.0,
    "reward_timeout": -8.0,
    "reward_turn_penalty": 0.01,
    "reward_repeat_position_penalty": 0.015,
    "reward_repeat_position_penalty_cap": 0.08,
    "reward_backtrack_penalty": 0.03,
    "reward_no_movement_penalty": 0.02,
    "exp_reward_norm_k": 100.0,
    "reward_exp_weight": 0.2,
    "reward_survival_alive_weight": 0.4,
    "reward_survival_hp_weight": 0.4,
    "reward_unit_upgrade": 0.25,
}

REQUIRED_WANDB_VERSION = (0, 25, 0)


def _wandb_version_tuple(raw_version: str) -> tuple[int, int, int] | None:
    """Parse semantic version (major.minor.patch) from a W&B version string."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", raw_version or "")
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def _wandb_version_is_supported() -> bool:
    """Require wandb>=0.25.0 for stable training integration."""
    if wandb is None:
        return False
    parsed = _wandb_version_tuple(getattr(wandb, "__version__", ""))
    return parsed is not None and parsed >= REQUIRED_WANDB_VERSION


def _normalize_wandb_entity(entity: str | None) -> str | None:
    """Normalize optional entity value from CLI/env."""
    if entity is None:
        return None
    cleaned = str(entity).strip()
    if not cleaned:
        return None
    if cleaned.lower() in {"none", "null", "default", "-"}:
        return None
    return cleaned


def _set_wandb_online_mode() -> None:
    """
    Force online mode for this process and best-effort update local wandb CLI mode.
    Avoids accidental sticky offline runs from previous sessions.
    """
    os.environ["WANDB_MODE"] = "online"
    try:
        subprocess.run(
            ["wandb", "online"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        # Non-fatal: env var still forces online mode for this process.
        pass


def _login_wandb_non_interactive() -> None:
    """
    Non-interactive W&B login.
    Uses WANDB_API_KEY when provided, otherwise relies on existing local credentials.
    Never prompts for input.
    """
    key = os.getenv("WANDB_API_KEY", "").strip()
    if not key:
        return
    wandb.login(key=key, relogin=False, verify=True)


def _init_wandb_run(project: str, name: str, config: dict, entity: str | None = None):
    """
    Initialize W&B strictly in online mode.
    Retries without entity on 403 because stale/invalid entity is a common cause.
    """
    _set_wandb_online_mode()
    normalized_entity = _normalize_wandb_entity(entity)

    init_kwargs = {
        "project": project,
        "name": name,
        "config": config,
        "sync_tensorboard": True,
        "save_code": True,
        "mode": "online",
    }

    if os.name == "nt":
        try:
            init_kwargs["settings"] = wandb.Settings(symlink=False)
        except Exception:
            # Keep env-var fallback when Settings API is unavailable.
            pass

    def _try_init(entity_value: str | None):
        kwargs = dict(init_kwargs)
        if entity_value:
            kwargs["entity"] = entity_value
        return wandb.init(**kwargs)

    last_exc: Exception | None = None

    # Attempt 1: requested/default entity.
    try:
        return _try_init(normalized_entity)
    except Exception as exc:
        last_exc = exc
        msg = str(exc)
        print(f"[WARN] W&B online init failed: {msg}")

        # Attempt 2: no explicit entity (uses account default).
        if normalized_entity and ("403" in msg or "permission" in msg.lower()):
            print("[WARN] Retrying W&B init without explicit entity...")
            try:
                return _try_init(None)
            except Exception as retry_exc:
                last_exc = retry_exc
                print(f"[WARN] Retry without entity failed: {retry_exc}")

    assert last_exc is not None
    raise RuntimeError(
        "W&B online init failed. "
        "Run `wandb login --relogin --verify` and ensure the API key has access "
        "to the selected project/entity."
    ) from last_exc


def mask_fn(env: CampaignEnv) -> np.ndarray:
    """Функция маскирования для ActionMasker."""
    return env.compute_action_mask()


def make_env(log_enabled: bool = False):
    """Фабрика создания среды."""
    base = CampaignEnv(
        grid_size=DEFAULT_GRID_SIZE,
        # Основная цель: пройти всю кампанию (4/4 врага).
        **REWARD_CONFIG,
        persist_blue_hp=True,
        log_enabled=log_enabled,
        max_grid_steps=1500,
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
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=os.getenv("WANDB_PROJECT", "disciples-campaign"),
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=os.getenv("WANDB_ENTITY", None),
        help="W&B entity (username or team)",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name")
    args = parser.parse_args()

    # Проверка среды
    print("Проверка среды CampaignEnv...")
    test_env = CampaignEnv(
        grid_size=DEFAULT_GRID_SIZE,
        log_enabled=False,
        realcapital=2,
        **REWARD_CONFIG,
    )
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
    if WANDB_AVAILABLE and not _wandb_version_is_supported():
        detected = getattr(wandb, "__version__", "unknown")
        required = ".".join(str(v) for v in REQUIRED_WANDB_VERSION)
        print(f"[WARN] wandb>={required} is required, found {detected}. W&B logging disabled.")
        WANDB_AVAILABLE = False

    run_name = args.run_name or f"campaign-ppo-{TOTAL_STEPS // 1000}k"

    if WANDB_AVAILABLE:
        try:
            _login_wandb_non_interactive()
        except Exception as exc:
            WANDB_AVAILABLE = False
            run_id = f"offline-{int(time.time())}"
            print(f"[WARN] W&B auth failed, training will continue without W&B: {exc}")

    if WANDB_AVAILABLE:
        wandb_config = {
            "algo": "MaskablePPO",
            "total_timesteps": TOTAL_STEPS,
            "n_envs": N_ENVS,
            "n_steps": 2048,
            "batch_size": 256,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "learning_rate": 3e-4,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "checkpoint_freq": CHECKPOINT_FREQ,
            "eval_freq": EVAL_FREQ,
            "grid_size": DEFAULT_GRID_SIZE,
            "num_enemies": len(test_env.grid_env.enemy_positions),
            "persist_blue_hp": True,
            **REWARD_CONFIG,
            "vecnormalize_norm_obs": VECNORM_NORM_OBS,
            "vecnormalize_norm_reward": VECNORM_NORM_REWARD,
            "vecnormalize_clip_reward": VECNORM_CLIP_REWARD,
        }
        try:
            run = _init_wandb_run(
                project=args.wandb_project,
                name=run_name,
                config=wandb_config,
                entity=args.wandb_entity,
            )
            run_id = run.id
            print(f"[INFO] W&B enabled (wandb={wandb.__version__}, mode={run.settings.mode})")
        except Exception as exc:
            WANDB_AVAILABLE = False
            run_id = f"offline-{int(time.time())}"
            print(f"[WARN] W&B init failed, training will continue without W&B: {exc}")
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
        gamma=0.995,
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
        gamma=0.995,
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
        gamma=0.995,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
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
