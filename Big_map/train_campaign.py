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
from collections import Counter, deque
from pathlib import Path
from typing import Any
import gymnasium as gym
import numpy as np
from console_encoding import setup_utf8_console

# Отключаем torch._dynamo, чтобы избежать зависимости от sympy/gmpy при обучении.
# Это стабилизирует запуск на окружениях без полной установки SymPy.
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
setup_utf8_console()

try:
    import comet_ml
    from comet_ml import Experiment, OfflineExperiment
except ImportError:
    comet_ml = None
    Experiment = None
    OfflineExperiment = None

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy as sb3_evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

try:
    from sb3_contrib.common.maskable.evaluation import (
        evaluate_policy as maskable_evaluate_policy,
    )
    _HAS_MASKABLE_EVAL = True
except ImportError:
    maskable_evaluate_policy = None
    _HAS_MASKABLE_EVAL = False

from campaign_env import CampaignEnv
from grid import DEFAULT_GRID_SIZE

# Директория этого файла (чтобы пути были стабильны независимо от cwd)
BASE_DIR = Path(__file__).resolve().parent
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
    "battle_reward_scale": 0.25,
    "battle_reward_win": 2.0,
    "battle_reward_loss": -1.0,
    "battle_reward_step": 0.01,
    "reward_battle_item_equip": 0.02,
    "reward_battle_item_use": 0.05,
    "reward_sell_junk_item": 0.05,
    "reward_spell_learn": 2.0,
    "reward_spell_cast": 0.005,
}

REQUIRED_COMET_VERSION = (3, 57, 0)
MAX_EVAL_EPISODE_STEPS = 4000


def _comet_version_tuple(raw_version: str) -> tuple[int, int, int] | None:
    """Parse semantic version (major.minor.patch) from a Comet version string."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", raw_version or "")
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def _comet_version_is_supported() -> bool:
    """Require comet_ml>=3.57.0 for stable training integration."""
    if comet_ml is None:
        return False
    parsed = _comet_version_tuple(getattr(comet_ml, "__version__", ""))
    return parsed is not None and parsed >= REQUIRED_COMET_VERSION


def _normalize_optional_name(value: str | None) -> str | None:
    """Normalize optional CLI/env value."""
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if cleaned.lower() in {"none", "null", "default", "-"}:
        return None
    return cleaned


def _init_comet_experiment(
    *,
    project: str,
    name: str,
    config: dict[str, Any],
    workspace: str | None = None,
    offline: bool = False,
    offline_directory: str | None = None,
):
    """Initialize Comet online or offline and log the base config."""
    normalized_workspace = _normalize_optional_name(workspace)
    api_key = os.getenv("COMET_API_KEY", "").strip() or None
    common_kwargs = {
        "project_name": project,
        "workspace": normalized_workspace,
        "log_code": True,
        "log_graph": True,
        "auto_param_logging": True,
        "auto_metric_logging": True,
        "auto_histogram_tensorboard_logging": True,
        "parse_args": False,
        "display_summary_level": 0,
    }

    if offline:
        experiment = OfflineExperiment(
            api_key=api_key,
            offline_directory=offline_directory,
            **common_kwargs,
        )
    else:
        if not api_key:
            raise RuntimeError(
                "COMET_API_KEY is not set. Provide it in the environment or use --comet-offline."
            )
        experiment = Experiment(
            api_key=api_key,
            **common_kwargs,
        )

    experiment.set_name(name)
    experiment.log_parameters(config)
    experiment.log_other("tracking_backend", "comet")
    experiment.add_tag("campaign")
    experiment.add_tag("maskable_ppo")
    return experiment


def _safe_mean(values) -> float:
    """Return float mean for a sequence-like container, or 0.0 when empty."""
    if not values:
        return 0.0
    return float(np.mean(list(values)))


def _safe_rate(numerator: int, denominator: int) -> float:
    """Return a safe fraction for counters."""
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _metric_token(value: Any) -> str:
    """Normalize dynamic metric suffixes to a predictable ASCII token."""
    token = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip()).strip("_").lower()
    return token or "unknown"


def mask_fn(env: CampaignEnv) -> np.ndarray:
    """Функция маскирования для ActionMasker."""
    return env.compute_action_mask()


class EvalStepCapWrapper(gym.Wrapper):
    """Hard-stop eval episodes after a fixed number of env steps."""

    def __init__(self, env: gym.Env, max_steps: int):
        super().__init__(env)
        self.max_steps = max(1, int(max_steps))
        self.eval_steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.eval_steps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.eval_steps += 1

        if not terminated and not truncated and self.eval_steps >= self.max_steps:
            info = dict(info)
            truncated = True
            info["campaign_result"] = "eval_timeout"
            info["eval_step_cap_hit"] = True
            info["eval_step_cap"] = self.max_steps

        return obs, reward, terminated, truncated, info


def make_env(log_enabled: bool = False, eval_max_episode_steps: int | None = None):
    """Фабрика создания среды."""
    base = CampaignEnv(
        grid_size=DEFAULT_GRID_SIZE,
        # Основная цель: пройти всю кампанию (4/4 врага).
        **REWARD_CONFIG,
        persist_blue_hp=True,
        log_enabled=log_enabled,
        max_grid_steps=1800,
        realcapital=2,
    )
    env = ActionMasker(base, mask_fn)
    if eval_max_episode_steps is not None:
        env = EvalStepCapWrapper(env, max_steps=eval_max_episode_steps)
    return Monitor(env)


class CampaignCheckpointCallback(BaseCallback):
    """Сохраняет модель каждые N шагов."""

    def __init__(
        self,
        save_dir: str,
        milestone_steps: int = 500_000,
        experiment: Any | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.milestone_steps = int(milestone_steps)
        self.experiment = experiment
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
            if self.experiment is not None:
                self.experiment.log_metric(
                    "checkpoint_saved_at",
                    self.next_milestone,
                    step=self.num_timesteps,
                )
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


def is_better_campaign_eval(
    *,
    victory_rate: float,
    mean_reward: float,
    best_victory_rate: float,
    best_mean_reward: float,
) -> bool:
    """Choose the best checkpoint by victory rate, using mean reward only as a tie-break."""
    victory_rate = float(victory_rate)
    mean_reward = float(mean_reward)
    best_victory_rate = float(best_victory_rate)
    best_mean_reward = float(best_mean_reward)

    if victory_rate > best_victory_rate:
        return True
    if np.isclose(victory_rate, best_victory_rate):
        return mean_reward > best_mean_reward
    return False


class CampaignVictoryEvalCallback(EvalCallback):
    """Eval callback that logs deterministic and stochastic campaign win rates.

    The best checkpoint is still chosen by deterministic victory rate.
    """

    def __init__(self, *args, use_masking: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_masking = bool(use_masking)
        self.best_victory_rate = -np.inf
        self.last_victory_rate = 0.0
        self.last_stochastic_victory_rate = 0.0
        self._eval_victory_buffer: list[bool] = []
        self.evaluations_victories: list[list[bool]] = []
        self.stochastic_evaluations_results: list[list[float]] = []
        self.stochastic_evaluations_length: list[list[int]] = []
        self.stochastic_evaluations_victories: list[list[bool]] = []
        self.stochastic_evaluations_successes: list[list[bool]] = []

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        super()._log_success_callback(locals_, globals_)
        if locals_["done"]:
            info = locals_["info"]
            self._eval_victory_buffer.append(info.get("campaign_result") == "victory")

    def _evaluate_policy(self, *, deterministic: bool):
        if self.use_masking:
            if maskable_evaluate_policy is None:
                raise RuntimeError("Maskable evaluation is unavailable, but use_masking=True.")
            return maskable_evaluate_policy(
                self.model,  # type: ignore[arg-type]
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
                use_masking=True,
            )
        return sb3_evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=deterministic,
            return_episode_rewards=True,
            warn=self.warn,
            callback=self._log_success_callback,
        )

    def _run_campaign_eval(self, *, deterministic: bool) -> tuple[list[float], list[int], list[bool], list[bool]]:
        self._is_success_buffer = []
        self._eval_victory_buffer = []
        episode_rewards, episode_lengths = self._evaluate_policy(deterministic=deterministic)
        return (
            list(episode_rewards),
            list(episode_lengths),
            list(self._is_success_buffer),
            list(self._eval_victory_buffer),
        )

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            det_rewards, det_lengths, det_successes, det_victories = self._run_campaign_eval(
                deterministic=True
            )
            stoch_rewards, stoch_lengths, stoch_successes, stoch_victories = self._run_campaign_eval(
                deterministic=False
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(det_rewards)
                self.evaluations_length.append(det_lengths)
                self.evaluations_victories.append(det_victories)
                self.stochastic_evaluations_results.append(stoch_rewards)
                self.stochastic_evaluations_length.append(stoch_lengths)
                self.stochastic_evaluations_victories.append(stoch_victories)

                kwargs = {
                    "victories": self.evaluations_victories,
                    "deterministic_results": self.evaluations_results,
                    "deterministic_ep_lengths": self.evaluations_length,
                    "deterministic_victories": self.evaluations_victories,
                    "stochastic_results": self.stochastic_evaluations_results,
                    "stochastic_ep_lengths": self.stochastic_evaluations_length,
                    "stochastic_victories": self.stochastic_evaluations_victories,
                }
                if det_successes:
                    self.evaluations_successes.append(det_successes)
                    kwargs["successes"] = self.evaluations_successes
                    kwargs["deterministic_successes"] = self.evaluations_successes
                if stoch_successes:
                    self.stochastic_evaluations_successes.append(stoch_successes)
                    kwargs["stochastic_successes"] = self.stochastic_evaluations_successes

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward = float(np.mean(det_rewards))
            std_reward = float(np.std(det_rewards))
            mean_ep_length = float(np.mean(det_lengths))
            std_ep_length = float(np.std(det_lengths))
            victory_rate = float(np.mean(det_victories)) if det_victories else 0.0
            stochastic_mean_reward = float(np.mean(stoch_rewards))
            stochastic_std_reward = float(np.std(stoch_rewards))
            stochastic_mean_ep_length = float(np.mean(stoch_lengths))
            stochastic_std_ep_length = float(np.std(stoch_lengths))
            stochastic_victory_rate = float(np.mean(stoch_victories)) if stoch_victories else 0.0
            self.last_mean_reward = mean_reward
            self.last_victory_rate = victory_rate
            self.last_stochastic_victory_rate = stochastic_victory_rate

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"det_reward={mean_reward:.2f} +/- {std_reward:.2f}, "
                    f"stoch_reward={stochastic_mean_reward:.2f} +/- {stochastic_std_reward:.2f}"
                )
                print(
                    f"Episode length: det={mean_ep_length:.2f} +/- {std_ep_length:.2f}, "
                    f"stoch={stochastic_mean_ep_length:.2f} +/- {stochastic_std_ep_length:.2f}"
                )
                print(
                    f"Victory rate: det={100 * victory_rate:.2f}%, "
                    f"stoch={100 * stochastic_victory_rate:.2f}%"
                )

            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/victory_rate", victory_rate)
            self.logger.record("eval/deterministic_mean_reward", mean_reward)
            self.logger.record("eval/deterministic_mean_ep_length", mean_ep_length)
            self.logger.record("eval/deterministic_victory_rate", victory_rate)
            self.logger.record("eval/stochastic_mean_reward", stochastic_mean_reward)
            self.logger.record("eval/stochastic_mean_ep_length", stochastic_mean_ep_length)
            self.logger.record("eval/stochastic_victory_rate", stochastic_victory_rate)

            if det_successes:
                success_rate = float(np.mean(det_successes))
                if self.verbose >= 1:
                    print(f"Success rate: det={100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)
                self.logger.record("eval/deterministic_success_rate", success_rate)
            if stoch_successes:
                stochastic_success_rate = float(np.mean(stoch_successes))
                if self.verbose >= 1:
                    print(f"Success rate (stochastic): {100 * stochastic_success_rate:.2f}%")
                self.logger.record("eval/stochastic_success_rate", stochastic_success_rate)

            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if is_better_campaign_eval(
                victory_rate=victory_rate,
                mean_reward=mean_reward,
                best_victory_rate=self.best_victory_rate,
                best_mean_reward=self.best_mean_reward,
            ):
                if self.verbose >= 1:
                    print("New best model by deterministic victory rate!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_victory_rate = victory_rate
                self.best_mean_reward = mean_reward
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class CampaignMetricsCallback(BaseCallback):
    """Логирует расширенные метрики кампании в Comet во время обучения."""

    def __init__(
        self,
        log_freq: int = 1000,
        episode_window: int = 100,
        experiment: Any | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_freq = max(1, int(log_freq))
        self.episode_window = max(1, int(episode_window))
        self.step_window = max(16, min(self.log_freq, 256))
        self.experiment = experiment

        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.recent_episode_rewards = deque(maxlen=self.episode_window)
        self.recent_episode_lengths = deque(maxlen=self.episode_window)
        self.recent_blue_hp_ratios = deque(maxlen=self.episode_window)
        self.recent_unit_upgrades = deque(maxlen=self.episode_window)
        self.recent_enemy_defeat_rewards = deque(maxlen=self.episode_window)
        self.recent_blue_exp_rewards = deque(maxlen=self.episode_window)
        self.recent_final_objective_rewards = deque(maxlen=self.episode_window)
        self.recent_castle_heal_uses = deque(maxlen=self.episode_window)
        self.recent_castle_heal_episode_flags = deque(maxlen=self.episode_window)
        self.recent_castle_healed_hp = deque(maxlen=self.episode_window)
        self.episode_chests_collected: list[float] = []
        self.recent_chests_collected = deque(maxlen=self.episode_window)
        self.recent_chest_episode_flags = deque(maxlen=self.episode_window)
        self.episode_ruins_cleared: list[float] = []
        self.recent_ruins_cleared = deque(maxlen=self.episode_window)
        self.recent_ruin_episode_flags = deque(maxlen=self.episode_window)
        self.episode_sold_items: list[float] = []
        self.recent_sold_items = deque(maxlen=self.episode_window)
        self.recent_sale_episode_flags = deque(maxlen=self.episode_window)
        self.recent_sale_gold = deque(maxlen=self.episode_window)
        self.episode_spells_learned: list[float] = []
        self.recent_spells_learned = deque(maxlen=self.episode_window)
        self.recent_spell_learning_episode_flags = deque(maxlen=self.episode_window)
        self.episode_spell_casts: list[float] = []
        self.recent_spell_casts = deque(maxlen=self.episode_window)
        self.recent_spell_cast_episode_flags = deque(maxlen=self.episode_window)
        self.episode_hired_units: list[float] = []
        self.recent_hired_units = deque(maxlen=self.episode_window)
        self.recent_hire_episode_flags = deque(maxlen=self.episode_window)
        self.episode_battle_items_equipped: list[float] = []
        self.recent_battle_items_equipped = deque(maxlen=self.episode_window)
        self.recent_battle_item_equip_episode_flags = deque(maxlen=self.episode_window)
        self.episode_battle_items_used: list[float] = []
        self.recent_battle_items_used = deque(maxlen=self.episode_window)
        self.recent_battle_item_use_episode_flags = deque(maxlen=self.episode_window)
        self.recent_turns = deque(maxlen=self.step_window)
        self.recent_gold = deque(maxlen=self.step_window)
        self.recent_moves = deque(maxlen=self.step_window)

        self.result_counter: Counter[str] = Counter()
        self.window_result_counter: Counter[str] = Counter()
        self.battle_counter: Counter[str] = Counter()
        self.window_battle_counter: Counter[str] = Counter()
        self.victory_reason_counter: Counter[str] = Counter()
        self.window_victory_reason_counter: Counter[str] = Counter()

        self.victories = 0
        self.defeats = 0
        self.timeouts = 0
        self.total_unit_upgrades = 0
        self.total_castle_heal_uses = 0
        self.total_castle_healed_hp = 0.0
        self.castle_heal_episodes = 0
        self.total_chests_collected = 0
        self.chest_collection_episodes = 0
        self.total_ruins_cleared = 0
        self.ruin_clear_episodes = 0
        self.total_sold_items = 0
        self.total_merchant_sale_gold = 0.0
        self.sale_episodes = 0
        self.total_spells_learned = 0
        self.spell_learning_episodes = 0
        self.total_spell_casts = 0
        self.spell_cast_episodes = 0
        self.total_hired_units = 0
        self.hire_episodes = 0
        self.total_battle_items_equipped = 0
        self.battle_item_equip_episodes = 0
        self.total_battle_items_used = 0
        self.battle_item_use_episodes = 0
        self.best_recent_victory_rate = 0.0
        self._last_logged_step = 0
        self._episode_castle_heal_uses: list[int] = []
        self._episode_castle_heal_flags: list[bool] = []
        self._episode_castle_healed_hp: list[float] = []
        self._episode_chests_collected: list[int] = []
        self._episode_ruins_cleared: list[int] = []
        self._episode_sold_items: list[int] = []
        self._episode_sale_gold: list[float] = []
        self._episode_spells_learned: list[int] = []
        self._episode_spell_casts: list[int] = []
        self._episode_hired_units: list[int] = []
        self._episode_battle_items_equipped: list[int] = []
        self._episode_battle_items_used: list[int] = []

    def _append_recent_stat(self, info: dict[str, Any], key: str, target: deque) -> None:
        value = info.get(key)
        if value is None:
            return
        try:
            target.append(float(value))
        except (TypeError, ValueError):
            return

    def _ensure_env_trackers(self, env_count: int) -> None:
        if env_count <= len(self._episode_castle_heal_uses):
            return
        missing = env_count - len(self._episode_castle_heal_uses)
        self._episode_castle_heal_uses.extend([0] * missing)
        self._episode_castle_heal_flags.extend([False] * missing)
        self._episode_castle_healed_hp.extend([0.0] * missing)
        self._episode_chests_collected.extend([0] * missing)
        self._episode_ruins_cleared.extend([0] * missing)
        self._episode_sold_items.extend([0] * missing)
        self._episode_sale_gold.extend([0.0] * missing)
        self._episode_spells_learned.extend([0] * missing)
        self._episode_spell_casts.extend([0] * missing)
        self._episode_hired_units.extend([0] * missing)
        self._episode_battle_items_equipped.extend([0] * missing)
        self._episode_battle_items_used.extend([0] * missing)

    def _consume_castle_heal(self, info: dict[str, Any], env_index: int) -> None:
        if not info.get("castle_heal_action"):
            return
        try:
            healed_amount = float(info.get("healed_amount", 0.0) or 0.0)
        except (TypeError, ValueError):
            healed_amount = 0.0
        if healed_amount <= 0.0:
            return

        self.total_castle_heal_uses += 1
        self.total_castle_healed_hp += healed_amount
        self._episode_castle_heal_uses[env_index] += 1
        self._episode_castle_healed_hp[env_index] += healed_amount
        self._episode_castle_heal_flags[env_index] = True

    def _finalize_episode_castle_heal(self, env_index: int) -> None:
        uses = float(self._episode_castle_heal_uses[env_index])
        healed_hp = float(self._episode_castle_healed_hp[env_index])
        used_flag = bool(self._episode_castle_heal_flags[env_index])

        self.recent_castle_heal_uses.append(uses)
        self.recent_castle_heal_episode_flags.append(1.0 if used_flag else 0.0)
        self.recent_castle_healed_hp.append(healed_hp)
        if used_flag:
            self.castle_heal_episodes += 1

        self._episode_castle_heal_uses[env_index] = 0
        self._episode_castle_heal_flags[env_index] = False
        self._episode_castle_healed_hp[env_index] = 0.0

    def _consume_chests(self, info: dict[str, Any], env_index: int) -> None:
        collected = info.get("collected_chests")
        if not isinstance(collected, (list, tuple)):
            return
        chest_count = len(collected)
        if chest_count <= 0:
            return
        self.total_chests_collected += chest_count
        self._episode_chests_collected[env_index] += chest_count

    def _finalize_episode_chests(self, env_index: int) -> None:
        chest_count = float(self._episode_chests_collected[env_index])
        self.episode_chests_collected.append(chest_count)
        self.recent_chests_collected.append(chest_count)
        self.recent_chest_episode_flags.append(1.0 if chest_count > 0.0 else 0.0)
        if chest_count > 0.0:
            self.chest_collection_episodes += 1

        if self.experiment is not None:
            self.experiment.log_metric(
                "campaign/episode_chests_collected",
                chest_count,
                step=self.num_timesteps,
            )

        self._episode_chests_collected[env_index] = 0

    def _consume_ruins(self, info: dict[str, Any], env_index: int) -> None:
        if not info.get("ruin_reward_applied"):
            return
        self.total_ruins_cleared += 1
        self._episode_ruins_cleared[env_index] += 1

    def _finalize_episode_ruins(self, env_index: int) -> None:
        ruin_count = float(self._episode_ruins_cleared[env_index])
        self.episode_ruins_cleared.append(ruin_count)
        self.recent_ruins_cleared.append(ruin_count)
        self.recent_ruin_episode_flags.append(1.0 if ruin_count > 0.0 else 0.0)
        if ruin_count > 0.0:
            self.ruin_clear_episodes += 1

        if self.experiment is not None:
            self.experiment.log_metric(
                "campaign/episode_ruins_cleared",
                ruin_count,
                step=self.num_timesteps,
            )

        self._episode_ruins_cleared[env_index] = 0

    def _consume_merchant_sales(self, info: dict[str, Any], env_index: int) -> None:
        try:
            sold_items = int(info.get("merchant_sold_items_count", 0) or 0)
        except (TypeError, ValueError):
            sold_items = 0
        try:
            sale_gold = float(info.get("merchant_sale_gold", 0.0) or 0.0)
        except (TypeError, ValueError):
            sale_gold = 0.0

        if sold_items <= 0 and sale_gold <= 0.0:
            return

        self.total_sold_items += max(0, sold_items)
        self.total_merchant_sale_gold += max(0.0, sale_gold)
        self._episode_sold_items[env_index] += max(0, sold_items)
        self._episode_sale_gold[env_index] += max(0.0, sale_gold)

    def _finalize_episode_sales(self, env_index: int) -> None:
        sold_items = float(self._episode_sold_items[env_index])
        sale_gold = float(self._episode_sale_gold[env_index])
        sold_flag = sold_items > 0.0 or sale_gold > 0.0

        self.episode_sold_items.append(sold_items)
        self.recent_sold_items.append(sold_items)
        self.recent_sale_episode_flags.append(1.0 if sold_flag else 0.0)
        self.recent_sale_gold.append(sale_gold)
        if sold_flag:
            self.sale_episodes += 1

        if self.experiment is not None:
            self.experiment.log_metric(
                "campaign/episode_sold_items",
                sold_items,
                step=self.num_timesteps,
            )
            self.experiment.log_metric(
                "campaign/episode_merchant_sale_gold",
                sale_gold,
                step=self.num_timesteps,
            )

        self._episode_sold_items[env_index] = 0
        self._episode_sale_gold[env_index] = 0.0

    def _consume_spell_learning(self, info: dict[str, Any], env_index: int) -> None:
        if not info.get("spell_learned"):
            return
        self.total_spells_learned += 1
        self._episode_spells_learned[env_index] += 1

    def _finalize_episode_spell_learning(self, env_index: int) -> None:
        spell_count = float(self._episode_spells_learned[env_index])
        learned_flag = spell_count > 0.0

        self.episode_spells_learned.append(spell_count)
        self.recent_spells_learned.append(spell_count)
        self.recent_spell_learning_episode_flags.append(1.0 if learned_flag else 0.0)
        if learned_flag:
            self.spell_learning_episodes += 1

        if self.experiment is not None:
            self.experiment.log_metric(
                "campaign/episode_spells_learned",
                spell_count,
                step=self.num_timesteps,
            )

        self._episode_spells_learned[env_index] = 0

    def _consume_spell_casts(self, info: dict[str, Any], env_index: int) -> None:
        if not info.get("spell_cast_action"):
            return
        if not info.get("spell_cast_executed"):
            return
        self.total_spell_casts += 1
        self._episode_spell_casts[env_index] += 1

    def _finalize_episode_spell_casts(self, env_index: int) -> None:
        cast_count = float(self._episode_spell_casts[env_index])
        cast_flag = cast_count > 0.0

        self.episode_spell_casts.append(cast_count)
        self.recent_spell_casts.append(cast_count)
        self.recent_spell_cast_episode_flags.append(1.0 if cast_flag else 0.0)
        if cast_flag:
            self.spell_cast_episodes += 1

        if self.experiment is not None:
            self.experiment.log_metric(
                "campaign/episode_spell_casts",
                cast_count,
                step=self.num_timesteps,
            )

        self._episode_spell_casts[env_index] = 0

    def _consume_hires(self, info: dict[str, Any], env_index: int) -> None:
        try:
            hired_units = int(info.get("hired_units_count", 0) or 0)
        except (TypeError, ValueError):
            hired_units = 0
        if hired_units <= 0 and info.get("hired"):
            hired_units = 1
        if hired_units <= 0:
            return

        self.total_hired_units += hired_units
        self._episode_hired_units[env_index] += hired_units

    def _finalize_episode_hires(self, env_index: int) -> None:
        hired_units = float(self._episode_hired_units[env_index])
        hired_flag = hired_units > 0.0

        self.episode_hired_units.append(hired_units)
        self.recent_hired_units.append(hired_units)
        self.recent_hire_episode_flags.append(1.0 if hired_flag else 0.0)
        if hired_flag:
            self.hire_episodes += 1

        if self.experiment is not None:
            self.experiment.log_metric(
                "campaign/episode_hired_units",
                hired_units,
                step=self.num_timesteps,
            )

        self._episode_hired_units[env_index] = 0

    def _consume_battle_item_activity(self, info: dict[str, Any], env_index: int) -> None:
        equipped = bool(info.get("equip_battle_item_action")) and bool(
            info.get("equip_battle_item_applied")
        )
        used = (
            bool(info.get("battle_hero_item_action"))
            and bool(info.get("battle_hero_item_applied"))
            and bool(info.get("battle_hero_item_consumed"))
        )

        if equipped:
            self.total_battle_items_equipped += 1
            self._episode_battle_items_equipped[env_index] += 1

        if used:
            self.total_battle_items_used += 1
            self._episode_battle_items_used[env_index] += 1

    def _finalize_episode_battle_item_activity(self, env_index: int) -> None:
        equipped_count = float(self._episode_battle_items_equipped[env_index])
        used_count = float(self._episode_battle_items_used[env_index])
        equipped_flag = equipped_count > 0.0
        used_flag = used_count > 0.0

        self.episode_battle_items_equipped.append(equipped_count)
        self.recent_battle_items_equipped.append(equipped_count)
        self.recent_battle_item_equip_episode_flags.append(1.0 if equipped_flag else 0.0)
        if equipped_flag:
            self.battle_item_equip_episodes += 1

        self.episode_battle_items_used.append(used_count)
        self.recent_battle_items_used.append(used_count)
        self.recent_battle_item_use_episode_flags.append(1.0 if used_flag else 0.0)
        if used_flag:
            self.battle_item_use_episodes += 1

        if self.experiment is not None:
            self.experiment.log_metric(
                "campaign/episode_battle_items_equipped",
                equipped_count,
                step=self.num_timesteps,
            )
            self.experiment.log_metric(
                "campaign/episode_battle_items_used",
                used_count,
                step=self.num_timesteps,
            )

        self._episode_battle_items_equipped[env_index] = 0
        self._episode_battle_items_used[env_index] = 0

    def _record_window_metrics(self, infos: list[dict[str, Any]]) -> None:
        turns = []
        gold = []
        moves = []
        for info in infos:
            if not isinstance(info, dict):
                continue
            if "turns" in info:
                try:
                    turns.append(float(info["turns"]))
                except (TypeError, ValueError):
                    pass
            if "gold" in info:
                try:
                    gold.append(float(info["gold"]))
                except (TypeError, ValueError):
                    pass
            if "moves" in info:
                try:
                    moves.append(float(info["moves"]))
                except (TypeError, ValueError):
                    pass

        if turns:
            self.recent_turns.append(float(np.mean(turns)))
        if gold:
            self.recent_gold.append(float(np.mean(gold)))
        if moves:
            self.recent_moves.append(float(np.mean(moves)))

    def _consume_info(self, info: dict[str, Any], env_index: int | None = None) -> None:
        if env_index is not None:
            self._consume_castle_heal(info, env_index)
            self._consume_chests(info, env_index)
            self._consume_ruins(info, env_index)
            self._consume_merchant_sales(info, env_index)
            self._consume_spell_learning(info, env_index)
            self._consume_spell_casts(info, env_index)
            self._consume_hires(info, env_index)
            self._consume_battle_item_activity(info, env_index)

        episode = info.get("episode")
        if isinstance(episode, dict):
            reward = episode.get("r")
            length = episode.get("l")
            if reward is not None:
                reward = float(reward)
                self.episode_rewards.append(reward)
                self.recent_episode_rewards.append(reward)
            if length is not None:
                length = int(length)
                self.episode_lengths.append(length)
                self.recent_episode_lengths.append(length)
            if env_index is not None:
                self._finalize_episode_castle_heal(env_index)
                self._finalize_episode_chests(env_index)
                self._finalize_episode_ruins(env_index)
                self._finalize_episode_sales(env_index)
                self._finalize_episode_spell_learning(env_index)
                self._finalize_episode_spell_casts(env_index)
                self._finalize_episode_hires(env_index)
                self._finalize_episode_battle_item_activity(env_index)

        campaign_result = info.get("campaign_result")
        if campaign_result:
            result_key = str(campaign_result)
            self.result_counter[result_key] += 1
            self.window_result_counter[result_key] += 1
            if result_key == "victory":
                self.victories += 1
            elif result_key == "defeat":
                self.defeats += 1
            elif result_key == "timeout":
                self.timeouts += 1

        battle_result = info.get("battle_result")
        if battle_result:
            battle_key = str(battle_result)
            self.battle_counter[battle_key] += 1
            self.window_battle_counter[battle_key] += 1

        victory_reason = info.get("campaign_victory_reason")
        if victory_reason:
            reason_key = str(victory_reason)
            self.victory_reason_counter[reason_key] += 1
            self.window_victory_reason_counter[reason_key] += 1

        if battle_result == "victory":
            self._append_recent_stat(info, "blue_hp_ratio", self.recent_blue_hp_ratios)
            self._append_recent_stat(info, "enemy_defeat_reward", self.recent_enemy_defeat_rewards)
            self._append_recent_stat(info, "blue_exp_reward", self.recent_blue_exp_rewards)
            unit_upgrades = info.get("unit_upgrades")
            if unit_upgrades is not None:
                try:
                    upgrade_count = int(unit_upgrades)
                except (TypeError, ValueError):
                    upgrade_count = 0
                self.total_unit_upgrades += upgrade_count
                self.recent_unit_upgrades.append(float(upgrade_count))
            self._append_recent_stat(info, "final_objective_reward", self.recent_final_objective_rewards)

    def _build_log_payload(self) -> dict[str, float]:
        total_episodes = int(sum(self.result_counter.values()))
        total_battles = int(sum(self.battle_counter.values()))
        window_episodes = int(sum(self.window_result_counter.values()))
        window_battles = int(sum(self.window_battle_counter.values()))

        recent_victory_rate = _safe_rate(
            self.window_result_counter.get("victory", 0),
            window_episodes,
        )
        self.best_recent_victory_rate = max(self.best_recent_victory_rate, recent_victory_rate)

        payload: dict[str, float] = {
            "campaign/episodes_total": total_episodes,
            "campaign/victories_total": float(self.victories),
            "campaign/defeats_total": float(self.defeats),
            "campaign/timeouts_total": float(self.timeouts),
            "campaign/victory_rate": _safe_rate(self.victories, total_episodes),
            "campaign/defeat_rate": _safe_rate(self.defeats, total_episodes),
            "campaign/timeout_rate": _safe_rate(self.timeouts, total_episodes),
            "campaign/battles_total": total_battles,
            "campaign/battle_victories_total": float(self.battle_counter.get("victory", 0)),
            "campaign/battle_defeats_total": float(self.battle_counter.get("defeat", 0)),
            "campaign/battle_victories_per_episode_mean": _safe_rate(
                self.battle_counter.get("victory", 0),
                total_episodes,
            ),
            "campaign/battle_win_rate": _safe_rate(
                self.battle_counter.get("victory", 0),
                total_battles,
            ),
            "campaign/unit_upgrades_total": float(self.total_unit_upgrades),
            "campaign/castle_heal_uses_total": float(self.total_castle_heal_uses),
            "campaign/castle_healed_hp_total": float(self.total_castle_healed_hp),
            "campaign/castle_heal_episodes_rate": _safe_rate(
                self.castle_heal_episodes,
                total_episodes,
            ),
            "campaign/chests_collected_total": float(self.total_chests_collected),
            "campaign/chests_collected_per_episode_mean": _safe_rate(
                self.total_chests_collected,
                total_episodes,
            ),
            "campaign/chest_collection_episodes_rate": _safe_rate(
                self.chest_collection_episodes,
                total_episodes,
            ),
            "campaign/ruins_cleared_total": float(self.total_ruins_cleared),
            "campaign/ruins_cleared_per_episode_mean": _safe_rate(
                self.total_ruins_cleared,
                total_episodes,
            ),
            "campaign/ruin_clear_episodes_rate": _safe_rate(
                self.ruin_clear_episodes,
                total_episodes,
            ),
            "campaign/sold_items_total": float(self.total_sold_items),
            "campaign/sold_items_per_episode_mean": _safe_rate(
                self.total_sold_items,
                total_episodes,
            ),
            "campaign/merchant_sale_gold_total": float(self.total_merchant_sale_gold),
            "campaign/merchant_sale_gold_per_episode_mean": _safe_rate(
                self.total_merchant_sale_gold,
                total_episodes,
            ),
            "campaign/sale_episodes_rate": _safe_rate(
                self.sale_episodes,
                total_episodes,
            ),
            "campaign/spells_learned_total": float(self.total_spells_learned),
            "campaign/spells_learned_per_episode_mean": _safe_rate(
                self.total_spells_learned,
                total_episodes,
            ),
            "campaign/spell_learning_episodes_rate": _safe_rate(
                self.spell_learning_episodes,
                total_episodes,
            ),
            "campaign/spell_casts_total": float(self.total_spell_casts),
            "campaign/spell_casts_per_episode_mean": _safe_rate(
                self.total_spell_casts,
                total_episodes,
            ),
            "campaign/spell_cast_episodes_rate": _safe_rate(
                self.spell_cast_episodes,
                total_episodes,
            ),
            "campaign/hired_units_total": float(self.total_hired_units),
            "campaign/hired_units_per_episode_mean": _safe_rate(
                self.total_hired_units,
                total_episodes,
            ),
            "campaign/hire_episodes_rate": _safe_rate(
                self.hire_episodes,
                total_episodes,
            ),
            "campaign/battle_items_equipped_total": float(self.total_battle_items_equipped),
            "campaign/battle_items_equipped_per_episode_mean": _safe_rate(
                self.total_battle_items_equipped,
                total_episodes,
            ),
            "campaign/battle_item_equip_episodes_rate": _safe_rate(
                self.battle_item_equip_episodes,
                total_episodes,
            ),
            "campaign/battle_items_used_total": float(self.total_battle_items_used),
            "campaign/battle_items_used_per_episode_mean": _safe_rate(
                self.total_battle_items_used,
                total_episodes,
            ),
            "campaign/battle_item_use_episodes_rate": _safe_rate(
                self.battle_item_use_episodes,
                total_episodes,
            ),
            "campaign/recent_episodes": float(len(self.recent_episode_rewards)),
            "campaign/recent_episode_reward_mean": _safe_mean(self.recent_episode_rewards),
            "campaign/recent_episode_length_mean": _safe_mean(self.recent_episode_lengths),
            "campaign/recent_blue_hp_ratio_mean": _safe_mean(self.recent_blue_hp_ratios),
            "campaign/recent_unit_upgrades_mean": _safe_mean(self.recent_unit_upgrades),
            "campaign/recent_enemy_defeat_reward_mean": _safe_mean(self.recent_enemy_defeat_rewards),
            "campaign/recent_blue_exp_reward_mean": _safe_mean(self.recent_blue_exp_rewards),
            "campaign/recent_final_objective_reward_mean": _safe_mean(self.recent_final_objective_rewards),
            "campaign/recent_castle_heal_uses_mean": _safe_mean(self.recent_castle_heal_uses),
            "campaign/recent_castle_heal_episodes_rate": _safe_mean(
                self.recent_castle_heal_episode_flags
            ),
            "campaign/recent_castle_healed_hp_mean": _safe_mean(self.recent_castle_healed_hp),
            "campaign/recent_chests_collected_mean": _safe_mean(self.recent_chests_collected),
            "campaign/recent_chest_collection_episodes_rate": _safe_mean(
                self.recent_chest_episode_flags
            ),
            "campaign/recent_ruins_cleared_mean": _safe_mean(self.recent_ruins_cleared),
            "campaign/recent_ruin_clear_episodes_rate": _safe_mean(
                self.recent_ruin_episode_flags
            ),
            "campaign/recent_sold_items_mean": _safe_mean(self.recent_sold_items),
            "campaign/recent_sale_episodes_rate": _safe_mean(
                self.recent_sale_episode_flags
            ),
            "campaign/recent_merchant_sale_gold_mean": _safe_mean(self.recent_sale_gold),
            "campaign/recent_spells_learned_mean": _safe_mean(self.recent_spells_learned),
            "campaign/recent_spell_learning_episodes_rate": _safe_mean(
                self.recent_spell_learning_episode_flags
            ),
            "campaign/recent_spell_casts_mean": _safe_mean(self.recent_spell_casts),
            "campaign/recent_spell_cast_episodes_rate": _safe_mean(
                self.recent_spell_cast_episode_flags
            ),
            "campaign/recent_hired_units_mean": _safe_mean(self.recent_hired_units),
            "campaign/recent_hire_episodes_rate": _safe_mean(
                self.recent_hire_episode_flags
            ),
            "campaign/recent_battle_items_equipped_mean": _safe_mean(
                self.recent_battle_items_equipped
            ),
            "campaign/recent_battle_item_equip_episodes_rate": _safe_mean(
                self.recent_battle_item_equip_episode_flags
            ),
            "campaign/recent_battle_items_used_mean": _safe_mean(
                self.recent_battle_items_used
            ),
            "campaign/recent_battle_item_use_episodes_rate": _safe_mean(
                self.recent_battle_item_use_episode_flags
            ),
            "campaign/recent_turns_mean": _safe_mean(self.recent_turns),
            "campaign/recent_gold_mean": _safe_mean(self.recent_gold),
            "campaign/recent_moves_mean": _safe_mean(self.recent_moves),
            "campaign/window/episodes": float(window_episodes),
            "campaign/window/victory_rate": recent_victory_rate,
            "campaign/window/defeat_rate": _safe_rate(
                self.window_result_counter.get("defeat", 0),
                window_episodes,
            ),
            "campaign/window/timeout_rate": _safe_rate(
                self.window_result_counter.get("timeout", 0),
                window_episodes,
            ),
            "campaign/window/battles": float(window_battles),
            "campaign/window/battle_win_rate": _safe_rate(
                self.window_battle_counter.get("victory", 0),
                window_battles,
            ),
            "campaign/window/chests_collected_mean": _safe_mean(self.recent_chests_collected),
            "campaign/window/ruins_cleared_mean": _safe_mean(self.recent_ruins_cleared),
            "campaign/window/sold_items_mean": _safe_mean(self.recent_sold_items),
            "campaign/window/spells_learned_mean": _safe_mean(self.recent_spells_learned),
            "campaign/window/spell_casts_mean": _safe_mean(self.recent_spell_casts),
            "campaign/window/hired_units_mean": _safe_mean(self.recent_hired_units),
            "campaign/window/battle_items_equipped_mean": _safe_mean(
                self.recent_battle_items_equipped
            ),
            "campaign/window/battle_items_used_mean": _safe_mean(
                self.recent_battle_items_used
            ),
            "campaign/best_recent_victory_rate": self.best_recent_victory_rate,
        }

        for reason, count in sorted(self.window_victory_reason_counter.items()):
            payload[f"campaign/window_victory_reasons/{_metric_token(reason)}"] = float(count)

        return payload

    def _reset_log_window(self) -> None:
        self.window_result_counter.clear()
        self.window_battle_counter.clear()
        self.window_victory_reason_counter.clear()

    def _log_to_experiment(self, *, force: bool = False) -> None:
        if not force and self.num_timesteps - self._last_logged_step < self.log_freq:
            return

        payload = self._build_log_payload()
        if self.experiment is not None:
            self.experiment.log_metrics(payload, step=self.num_timesteps)
        self._last_logged_step = self.num_timesteps

        if self.experiment is None:
            self._reset_log_window()
            return

        self.experiment.log_other(
            "summary_campaign_episodes_total",
            int(sum(self.result_counter.values())),
        )
        self.experiment.log_other(
            "summary_campaign_victory_rate",
            payload["campaign/victory_rate"],
        )
        self.experiment.log_other(
            "summary_campaign_battle_win_rate",
            payload["campaign/battle_win_rate"],
        )
        self.experiment.log_other(
            "summary_campaign_best_recent_victory_rate",
            self.best_recent_victory_rate,
        )
        self._reset_log_window()

    def _on_step(self) -> bool:
        infos = [info for info in self.locals.get("infos", []) if isinstance(info, dict)]
        self._record_window_metrics(infos)
        self._ensure_env_trackers(len(infos))
        for env_index, info in enumerate(infos):
            self._consume_info(info, env_index=env_index)
        self._log_to_experiment()
        return True

    def _on_training_end(self) -> None:
        self._log_to_experiment(force=True)

    def save_chests_per_episode_plot(self, path: str | os.PathLike[str]) -> Path | None:
        if not self.episode_chests_collected:
            return None

        import matplotlib.pyplot as plt

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        episodes = np.arange(1, len(self.episode_chests_collected) + 1)
        chest_counts = np.asarray(self.episode_chests_collected, dtype=float)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(episodes, chest_counts, color="#8B5A2B", linewidth=1.6, label="Chests per episode")
        ax.fill_between(episodes, chest_counts, color="#D9B36C", alpha=0.35)

        if len(chest_counts) >= 5:
            window = min(25, len(chest_counts))
            kernel = np.ones(window, dtype=float) / float(window)
            rolling = np.convolve(chest_counts, kernel, mode="valid")
            rolling_x = episodes[window - 1 :]
            ax.plot(
                rolling_x,
                rolling,
                color="#2F4858",
                linewidth=2.0,
                label=f"Rolling mean ({window})",
            )

        ax.set_title("Collected Chests Per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Chests")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(target, dpi=160)
        plt.close(fig)
        return target

    def save_sold_items_per_episode_plot(self, path: str | os.PathLike[str]) -> Path | None:
        if not self.episode_sold_items:
            return None

        import matplotlib.pyplot as plt

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        episodes = np.arange(1, len(self.episode_sold_items) + 1)
        sold_counts = np.asarray(self.episode_sold_items, dtype=float)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(episodes, sold_counts, color="#8C1C13", linewidth=1.6, label="Sold items per episode")
        ax.fill_between(episodes, sold_counts, color="#D95D39", alpha=0.28)

        if len(sold_counts) >= 5:
            window = min(25, len(sold_counts))
            kernel = np.ones(window, dtype=float) / float(window)
            rolling = np.convolve(sold_counts, kernel, mode="valid")
            rolling_x = episodes[window - 1 :]
            ax.plot(
                rolling_x,
                rolling,
                color="#2F4858",
                linewidth=2.0,
                label=f"Rolling mean ({window})",
            )

        ax.set_title("Sold Merchant Items Per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Items sold")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(target, dpi=160)
        plt.close(fig)
        return target

    def save_ruins_cleared_per_episode_plot(self, path: str | os.PathLike[str]) -> Path | None:
        if not self.episode_ruins_cleared:
            return None

        import matplotlib.pyplot as plt

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        episodes = np.arange(1, len(self.episode_ruins_cleared) + 1)
        ruin_counts = np.asarray(self.episode_ruins_cleared, dtype=float)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(episodes, ruin_counts, color="#7F1D1D", linewidth=1.7, label="Ruins cleared per episode")
        ax.fill_between(episodes, ruin_counts, color="#DC2626", alpha=0.24)

        if len(ruin_counts) >= 5:
            window = min(25, len(ruin_counts))
            kernel = np.ones(window, dtype=float) / float(window)
            rolling = np.convolve(ruin_counts, kernel, mode="valid")
            rolling_x = episodes[window - 1 :]
            ax.plot(
                rolling_x,
                rolling,
                color="#1F2937",
                linewidth=2.0,
                label=f"Rolling mean ({window})",
            )

        ax.set_title("Cleared Ruins Per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Ruins")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(target, dpi=160)
        plt.close(fig)
        return target

    def save_spells_learned_per_episode_plot(self, path: str | os.PathLike[str]) -> Path | None:
        if not self.episode_spells_learned:
            return None

        import matplotlib.pyplot as plt

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        episodes = np.arange(1, len(self.episode_spells_learned) + 1)
        spell_counts = np.asarray(self.episode_spells_learned, dtype=float)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(episodes, spell_counts, color="#1D4ED8", linewidth=1.7, label="Spells learned per episode")
        ax.fill_between(episodes, spell_counts, color="#60A5FA", alpha=0.24)

        if len(spell_counts) >= 5:
            window = min(25, len(spell_counts))
            kernel = np.ones(window, dtype=float) / float(window)
            rolling = np.convolve(spell_counts, kernel, mode="valid")
            rolling_x = episodes[window - 1 :]
            ax.plot(
                rolling_x,
                rolling,
                color="#0F172A",
                linewidth=2.0,
                label=f"Rolling mean ({window})",
            )

        ax.set_title("Learned Spells Per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Spells")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(target, dpi=160)
        plt.close(fig)
        return target

    def save_spell_cast_activity_plot(self, path: str | os.PathLike[str]) -> Path | None:
        if not self.episode_spell_casts:
            return None

        import matplotlib.pyplot as plt

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        episodes = np.arange(1, len(self.episode_spell_casts) + 1)
        cast_counts = np.asarray(self.episode_spell_casts, dtype=float)
        cumulative_casts = np.cumsum(cast_counts)

        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 8.0), sharex=True)

        ax_top.plot(
            episodes,
            cast_counts,
            color="#B45309",
            linewidth=1.8,
            label="Spell casts per episode",
        )
        ax_top.fill_between(episodes, cast_counts, color="#FCD34D", alpha=0.28)
        if len(cast_counts) >= 5:
            window = min(25, len(cast_counts))
            kernel = np.ones(window, dtype=float) / float(window)
            rolling = np.convolve(cast_counts, kernel, mode="valid")
            rolling_x = episodes[window - 1 :]
            ax_top.plot(
                rolling_x,
                rolling,
                color="#78350F",
                linewidth=2.0,
                linestyle="--",
                label=f"Rolling mean ({window})",
            )
        ax_top.set_title("Spell Cast Usage")
        ax_top.set_ylabel("Casts per episode")
        ax_top.grid(True, alpha=0.25, linewidth=0.6)
        ax_top.legend(loc="upper right")

        ax_bottom.plot(
            episodes,
            cumulative_casts,
            color="#1D4ED8",
            linewidth=2.1,
            label="Cumulative spell casts",
        )
        ax_bottom.fill_between(episodes, cumulative_casts, color="#93C5FD", alpha=0.22)
        ax_bottom.set_xlabel("Episode")
        ax_bottom.set_ylabel("Cumulative casts")
        ax_bottom.grid(True, alpha=0.25, linewidth=0.6)
        ax_bottom.legend(loc="upper left")

        fig.tight_layout()
        fig.savefig(target, dpi=160)
        plt.close(fig)
        return target

    def save_hired_units_per_episode_plot(self, path: str | os.PathLike[str]) -> Path | None:
        if not self.episode_hired_units:
            return None

        import matplotlib.pyplot as plt

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        episodes = np.arange(1, len(self.episode_hired_units) + 1)
        hired_counts = np.asarray(self.episode_hired_units, dtype=float)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(episodes, hired_counts, color="#9A3412", linewidth=1.8, label="Hired units per episode")
        ax.fill_between(episodes, hired_counts, color="#FDBA74", alpha=0.3)

        if len(hired_counts) >= 5:
            window = min(25, len(hired_counts))
            kernel = np.ones(window, dtype=float) / float(window)
            rolling = np.convolve(hired_counts, kernel, mode="valid")
            rolling_x = episodes[window - 1 :]
            ax.plot(
                rolling_x,
                rolling,
                color="#1F2937",
                linewidth=2.0,
                label=f"Rolling mean ({window})",
            )

        ax.set_title("Hired Units Per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Units hired")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(target, dpi=160)
        plt.close(fig)
        return target

    def save_battle_item_activity_plot(self, path: str | os.PathLike[str]) -> Path | None:
        if not self.episode_battle_items_equipped and not self.episode_battle_items_used:
            return None

        import matplotlib.pyplot as plt

        equip_count = len(self.episode_battle_items_equipped)
        use_count = len(self.episode_battle_items_used)
        episode_count = max(equip_count, use_count)
        if episode_count <= 0:
            return None

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        episodes = np.arange(1, episode_count + 1)
        equipped_counts = np.asarray(
            list(self.episode_battle_items_equipped) + [0.0] * (episode_count - equip_count),
            dtype=float,
        )
        used_counts = np.asarray(
            list(self.episode_battle_items_used) + [0.0] * (episode_count - use_count),
            dtype=float,
        )

        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(
            episodes,
            equipped_counts,
            color="#0F766E",
            linewidth=1.8,
            label="Equipped per episode",
        )
        ax.fill_between(episodes, equipped_counts, color="#5EEAD4", alpha=0.22)
        ax.plot(
            episodes,
            used_counts,
            color="#7C3AED",
            linewidth=1.8,
            label="Used in battle per episode",
        )
        ax.fill_between(episodes, used_counts, color="#C4B5FD", alpha=0.18)

        if episode_count >= 5:
            window = min(25, episode_count)
            kernel = np.ones(window, dtype=float) / float(window)
            equipped_rolling = np.convolve(equipped_counts, kernel, mode="valid")
            used_rolling = np.convolve(used_counts, kernel, mode="valid")
            rolling_x = episodes[window - 1 :]
            ax.plot(
                rolling_x,
                equipped_rolling,
                color="#115E59",
                linewidth=2.0,
                linestyle="--",
                label=f"Equip rolling mean ({window})",
            )
            ax.plot(
                rolling_x,
                used_rolling,
                color="#5B21B6",
                linewidth=2.0,
                linestyle="--",
                label=f"Use rolling mean ({window})",
            )

        ax.set_title("Battle Item Equips and Uses Per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(target, dpi=160)
        plt.close(fig)
        return target

    def save_cumulative_hired_units_plot(self, path: str | os.PathLike[str]) -> Path | None:
        if not self.episode_hired_units:
            return None

        import matplotlib.pyplot as plt

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        episodes = np.arange(1, len(self.episode_hired_units) + 1)
        cumulative_hired = np.cumsum(np.asarray(self.episode_hired_units, dtype=float))

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(
            episodes,
            cumulative_hired,
            color="#0F766E",
            linewidth=2.1,
            label="Cumulative hired units",
        )
        ax.fill_between(episodes, cumulative_hired, color="#5EEAD4", alpha=0.24)
        ax.set_title("Cumulative Hired Units Over Training")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total units hired")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(target, dpi=160)
        plt.close(fig)
        return target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Campaign Agent")
    parser.add_argument("--total-steps", type=int, default=2_000_000, help="Total training steps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--checkpoint-freq", type=int, default=500_000, help="Checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Evaluation frequency")
    parser.add_argument("--no-comet", action="store_true", help="Disable Comet logging")
    parser.add_argument(
        "--comet-project",
        type=str,
        default=os.getenv("COMET_PROJECT_NAME", "disciples-campaign"),
        help="Comet project name",
    )
    parser.add_argument(
        "--comet-workspace",
        type=str,
        default=os.getenv("COMET_WORKSPACE", None),
        help="Comet workspace",
    )
    parser.add_argument(
        "--comet-offline",
        action="store_true",
        help="Use Comet OfflineExperiment instead of online logging",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name")
    parser.add_argument(
        "--comet-log-freq",
        type=int,
        default=1000,
        help="How often to send custom campaign metrics to Comet",
    )
    parser.add_argument(
        "--comet-window-episodes",
        type=int,
        default=100,
        help="Rolling episode window used for custom Comet analytics",
    )
    parser.add_argument(
        "--comet-offline-dir",
        type=str,
        default=str(BASE_DIR / "comet_offline"),
        help="Directory for Comet offline experiment files",
    )
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
    EVAL_EPISODES = 20
    VECNORM_NORM_OBS = False
    VECNORM_NORM_REWARD = True
    VECNORM_CLIP_REWARD = 10.0

    # -------------------- Инициализация Comet --------------------
    COMET_AVAILABLE = Experiment is not None and OfflineExperiment is not None and not args.no_comet
    if COMET_AVAILABLE and not _comet_version_is_supported():
        detected = getattr(comet_ml, "__version__", "unknown")
        required = ".".join(str(v) for v in REQUIRED_COMET_VERSION)
        print(f"[WARN] comet_ml>={required} is required, found {detected}. Comet logging disabled.")
        COMET_AVAILABLE = False

    run_name = args.run_name or f"campaign-ppo-{TOTAL_STEPS // 1000}k"

    comet_experiment = None
    comet_config = {
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
            "comet_log_freq": args.comet_log_freq,
            "eval_episodes": EVAL_EPISODES,
            "max_eval_episode_steps": MAX_EVAL_EPISODE_STEPS,
            "comet_window_episodes": args.comet_window_episodes,
            "comet_offline": bool(args.comet_offline),
            "grid_size": DEFAULT_GRID_SIZE,
            "num_enemies": len(test_env.grid_env.enemy_positions),
            "persist_blue_hp": True,
            **REWARD_CONFIG,
            "vecnormalize_norm_obs": VECNORM_NORM_OBS,
            "vecnormalize_norm_reward": VECNORM_NORM_REWARD,
            "vecnormalize_clip_reward": VECNORM_CLIP_REWARD,
        }

    if COMET_AVAILABLE:
        try:
            comet_experiment = _init_comet_experiment(
                project=args.comet_project,
                name=run_name,
                config=comet_config,
                workspace=args.comet_workspace,
                offline=args.comet_offline,
                offline_directory=args.comet_offline_dir,
            )
            run_id = comet_experiment.get_key()
            mode = "offline" if args.comet_offline else "online"
            print(f"[INFO] Comet enabled (comet_ml={comet_ml.__version__}, mode={mode})")
            if getattr(comet_experiment, "url", None):
                print(f"[INFO] Comet run: {comet_experiment.url}")
        except Exception as exc:
            COMET_AVAILABLE = False
            run_id = f"offline-{int(time.time())}"
            print(f"[WARN] Comet init failed, training will continue without Comet: {exc}")
    else:
        run_id = f"offline-{int(time.time())}"
        print("[INFO] Comet disabled, running without experiment tracking")

    # Директории
    CHECKPOINT_DIR = str(BASE_DIR / "campaign_checkpoints" / run_id)
    MODEL_DIR = f"./campaign_models/{run_id}"
    EVAL_DIR = f"./campaign_eval/{run_id}"
    TB_LOG_DIR = f"./campaign_tb_logs/{run_id}"
    COMET_OFFLINE_DIR = args.comet_offline_dir

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)
    os.makedirs(COMET_OFFLINE_DIR, exist_ok=True)

    if COMET_AVAILABLE and comet_experiment is not None:
        comet_experiment.log_parameters(
            {
                "run_id": run_id,
                "checkpoint_dir": CHECKPOINT_DIR,
                "model_dir": MODEL_DIR,
                "eval_dir": EVAL_DIR,
                "tb_log_dir": TB_LOG_DIR,
                "comet_offline_dir": COMET_OFFLINE_DIR,
            },
            prefix="paths",
        )

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
    eval_env = DummyVecEnv(
        [
            lambda: make_env(
                log_enabled=False,
                eval_max_episode_steps=MAX_EVAL_EPISODE_STEPS,
            )
        ]
    )
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
        device="cpu",
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
        experiment=comet_experiment,
        verbose=1,
    )
    callbacks.append(checkpoint_cb)

    # Метрики кампании
    metrics_cb = CampaignMetricsCallback(
        log_freq=args.comet_log_freq,
        episode_window=args.comet_window_episodes,
        experiment=comet_experiment,
        verbose=0,
    )
    callbacks.append(metrics_cb)

    # Evaluation callback
    best_vecnorm_cb = SaveVecNormalizeOnBestCallback(
        save_path=f"{MODEL_DIR}/best",
        verbose=0,
    )
    eval_cb = CampaignVictoryEvalCallback(
        eval_env,
        best_model_save_path=f"{MODEL_DIR}/best",
        log_path=EVAL_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False,
        callback_on_new_best=best_vecnorm_cb,
        use_masking=_HAS_MASKABLE_EVAL,
    )
    callbacks.append(eval_cb)

    # -------------------- Обучение --------------------
    print(f"\n{'='*60}")
    print("Запуск обучения Campaign Agent")
    print(f"Всего шагов: {TOTAL_STEPS:,}")
    print(f"Параллельных сред: {N_ENVS}")
    print(f"Checkpoint каждые: {CHECKPOINT_FREQ:,} шагов")
    print(f"Evaluation каждые: {EVAL_FREQ:,} шагов")
    print(f"Eval эпизодов на режим: {EVAL_EPISODES}")
    print(f"Eval лимит шагов на эпизод: {MAX_EVAL_EPISODE_STEPS}")
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

    hired_episode_plot_path = BASE_DIR / "outputs" / "campaign_hired_units_per_episode.png"
    saved_hired_episode_plot = metrics_cb.save_hired_units_per_episode_plot(hired_episode_plot_path)
    if saved_hired_episode_plot is not None:
        print(f"  График найма за эпизод: {saved_hired_episode_plot}")

    hired_total_plot_path = BASE_DIR / "outputs" / "campaign_hired_units_cumulative.png"
    saved_hired_total_plot = metrics_cb.save_cumulative_hired_units_plot(hired_total_plot_path)
    if saved_hired_total_plot is not None:
        print(f"  График найма за всё обучение: {saved_hired_total_plot}")

    battle_item_plot_path = BASE_DIR / "outputs" / "campaign_battle_item_activity.png"
    saved_battle_item_plot = metrics_cb.save_battle_item_activity_plot(battle_item_plot_path)
    if saved_battle_item_plot is not None:
        print(f"  График боевых предметов: {saved_battle_item_plot}")

    spell_cast_plot_path = BASE_DIR / "outputs" / "campaign_spell_cast_activity.png"
    saved_spell_cast_plot = metrics_cb.save_spell_cast_activity_plot(spell_cast_plot_path)
    if saved_spell_cast_plot is not None:
        print(f"  График применения заклинаний: {saved_spell_cast_plot}")

    if COMET_AVAILABLE and comet_experiment is not None:
        try:
            comet_experiment.log_tensorboard_folder(TB_LOG_DIR)
        except Exception as exc:
            print(f"[WARN] Failed to upload TensorBoard logs to Comet: {exc}")
        if saved_hired_episode_plot is not None and hasattr(comet_experiment, "log_image"):
            try:
                comet_experiment.log_image(
                    str(saved_hired_episode_plot),
                    name="campaign_hired_units_per_episode",
                )
            except Exception as exc:
                print(f"[WARN] Failed to upload hire-per-episode plot to Comet: {exc}")
        if saved_hired_total_plot is not None and hasattr(comet_experiment, "log_image"):
            try:
                comet_experiment.log_image(
                    str(saved_hired_total_plot),
                    name="campaign_hired_units_cumulative",
                )
            except Exception as exc:
                print(f"[WARN] Failed to upload cumulative-hire plot to Comet: {exc}")
        if saved_battle_item_plot is not None and hasattr(comet_experiment, "log_image"):
            try:
                comet_experiment.log_image(
                    str(saved_battle_item_plot),
                    name="campaign_battle_item_activity",
                )
            except Exception as exc:
                print(f"[WARN] Failed to upload battle-item plot to Comet: {exc}")
        if saved_spell_cast_plot is not None and hasattr(comet_experiment, "log_image"):
            try:
                comet_experiment.log_image(
                    str(saved_spell_cast_plot),
                    name="campaign_spell_cast_activity",
                )
            except Exception as exc:
                print(f"[WARN] Failed to upload spell-cast plot to Comet: {exc}")
        try:
            comet_experiment.log_other(
                "final_campaign_victories",
                int(metrics_cb.victories),
            )
            comet_experiment.log_other(
                "final_campaign_defeats",
                int(metrics_cb.defeats),
            )
            comet_experiment.log_other(
                "final_campaign_timeouts",
                int(metrics_cb.timeouts),
            )
        finally:
            comet_experiment.end()
