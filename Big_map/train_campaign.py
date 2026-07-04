import os
import re
import time
import traceback
import json
from collections import Counter, deque
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gymnasium as gym
import numpy as np
import torch as th
from console_encoding import setup_utf8_console

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
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecCheckNan,
    VecNormalize,
    sync_envs_normalization,
)

from sb3_contrib.ppo_mask import MaskablePPO

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


def _patch_maskable_categorical_cached_probs_validation() -> None:
    try:
        from sb3_contrib.common.maskable.distributions import MaskableCategorical
        from torch.distributions import Categorical
        from torch.distributions.utils import logits_to_probs
    except ImportError:
        return

    current_apply_masking = getattr(MaskableCategorical, "apply_masking", None)
    if getattr(current_apply_masking, "_campaign_cache_safe_patch", False):
        return

    def apply_masking(self, masks) -> None:
        if masks is not None:
            device = self.logits.device
            self.masks = th.as_tensor(masks, dtype=th.bool, device=device).reshape(
                self.logits.shape
            )
            huge_neg = th.tensor(-1e8, dtype=self.logits.dtype, device=device)
            logits = th.where(self.masks, self._original_logits, huge_neg)
        else:
            self.masks = None
            logits = self._original_logits

        self.__dict__.pop("probs", None)
        self.__dict__.pop("logits", None)
        Categorical.__init__(self, logits=logits, validate_args=False)
        self.probs = logits_to_probs(self.logits)

    apply_masking._campaign_cache_safe_patch = True
    MaskableCategorical.apply_masking = apply_masking


_patch_maskable_categorical_cached_probs_validation()

BASE_DIR = Path(__file__).resolve().parent

REWARD_CONFIG = {
    "reward_engage_battle": 0.25,
    "reward_defeat_enemy": 4.0,
    "reward_all_enemies": 8.0,
    "reward_objective_city_reached": 6.0,
    "reward_green_dragon_reached": 6.0,
    "reward_green_dragon_objective_multiplier": 2.0,
    "reward_dragon_damage_fraction": 0.0,
    "reward_dragon_rounds_record": 0.0,
    "reward_no_dragon_victory_late_step_penalty": 0.004,
    "reward_no_dragon_victory_late_step_penalty_cap": 0.02,
    "reward_no_dragon_victory_late_step_start_fraction": 0.5,
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
    "reward_unit_upgrade": 3.0,
    "reward_unit_tier_bonus": 0.5,
    "reward_enemies_defeated_weight": 5.0,
    "battle_reward_scale": 0.25,
    "battle_reward_win": 2.0,
    "battle_reward_loss": -1.0,
    "battle_reward_step": 0.01,
    "reward_battle_item_equip": 0.02,
    "reward_battle_item_use": 0.05,
    "reward_sell_junk_item": 0.05,
    "reward_unit_swap_penalty": 0.004,
    "reward_spell_learn": 2.0,
    "reward_spell_cast": 0.005,
}

REQUIRED_COMET_VERSION = (3, 57, 0)
MAX_EVAL_EPISODE_STEPS = 5000
# Жёсткий потолок полной длины эпизода (все env-шаги, карта + бои) в обучении.
TRAIN_MAX_EPISODE_STEPS = 5000


def _comet_version_tuple(raw_version: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", raw_version or "")
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def _comet_version_is_supported() -> bool:
    if comet_ml is None:
        return False
    parsed = _comet_version_tuple(getattr(comet_ml, "__version__", ""))
    return parsed is not None and parsed >= REQUIRED_COMET_VERSION


def _normalize_optional_name(value: str | None) -> str | None:
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
    if not values:
        return 0.0
    return float(np.mean(list(values)))


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


class EvalStepCapWrapper(gym.Wrapper):

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


def make_env(
    log_enabled: bool = False,
    eval_max_episode_steps: int | None = None,
    freeze_dynamic_action_layout: bool = True,
    scripted_capital_bot_enabled: bool = True,
    empire_territory_enabled: bool = True,
    campaign_objective: str = "cities",
):
    env = CampaignEnv(
        grid_size=DEFAULT_GRID_SIZE,
        **REWARD_CONFIG,
        persist_blue_hp=True,
        log_enabled=log_enabled,
        detailed_step_info=bool(log_enabled),
        freeze_dynamic_action_layout=freeze_dynamic_action_layout,
        scripted_capital_bot_enabled=scripted_capital_bot_enabled,
        empire_territory_enabled=empire_territory_enabled,
        campaign_objective=campaign_objective,
        max_grid_steps=3000,
        Realcapital=2,
    )
    # Жёсткий потолок полной длины эпизода: в eval — eval_max_episode_steps,
    # в обучении — TRAIN_MAX_EPISODE_STEPS (гарантирует завершение даже при
    # действиях, не двигающих grid step_count: касты/стройка и т.п.).
    step_cap = eval_max_episode_steps if eval_max_episode_steps is not None else TRAIN_MAX_EPISODE_STEPS
    env = EvalStepCapWrapper(env, max_steps=step_cap)
    return Monitor(env)


def resolve_training_vec_env_config(vec_env_mode: str):
    if vec_env_mode == "subproc":
        return SubprocVecEnv, {"start_method": "spawn"}
    if vec_env_mode == "dummy":
        return DummyVecEnv, {}
    raise ValueError(f"Unsupported vec env mode: {vec_env_mode!r}")


def _to_numpy_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if th.is_tensor(value):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None


def _slice_rollout_array(value: Any, used_size: int) -> np.ndarray | None:
    array = _to_numpy_array(value)
    if array is None:
        return None
    if array.ndim > 0 and array.shape[0] >= used_size:
        return array[:used_size]
    return array


def _format_array_summary(name: str, value: Any, *, used_size: int | None = None) -> list[str]:
    array = _to_numpy_array(value)
    if array is None:
        return [f"{name}: unavailable"]
    if used_size is not None and array.ndim > 0 and array.shape[0] >= used_size:
        array = array[:used_size]

    lines = [f"{name}: shape={tuple(array.shape)}, dtype={array.dtype}, size={int(array.size)}"]
    if array.size <= 0:
        return lines

    if np.issubdtype(array.dtype, np.bool_):
        true_count = int(array.sum())
        lines.append(f"{name}: true={true_count}, false={int(array.size) - true_count}")
        if array.ndim >= 1 and array.shape[-1] > 0:
            row_view = array.reshape(-1, array.shape[-1])
            zero_rows = int(np.sum(row_view.sum(axis=1) == 0))
            lines.append(f"{name}: zero_valid_action_rows={zero_rows}")
        return lines

    if np.issubdtype(array.dtype, np.number):
        finite_mask = np.isfinite(array)
        nan_count = int(np.isnan(array).sum()) if np.issubdtype(array.dtype, np.inexact) else 0
        inf_count = int(np.isinf(array).sum()) if np.issubdtype(array.dtype, np.inexact) else 0
        finite_count = int(finite_mask.sum())
        lines.append(
            f"{name}: finite={finite_count}/{int(array.size)}, nan={nan_count}, inf={inf_count}"
        )
        if finite_count > 0:
            finite_values = array[finite_mask]
            lines.append(
                f"{name}: min={float(finite_values.min()):.6g}, "
                f"max={float(finite_values.max()):.6g}, "
                f"mean={float(finite_values.mean()):.6g}"
            )
        return lines

    return lines


def _rollout_buffer_used_size(model: Any) -> int:
    rollout_buffer = getattr(model, "rollout_buffer", None)
    if rollout_buffer is None:
        return 0
    if bool(getattr(rollout_buffer, "full", False)):
        return int(getattr(rollout_buffer, "buffer_size", 0) or 0)
    return int(getattr(rollout_buffer, "pos", 0) or 0)


def _collect_policy_parameter_issues(model: Any) -> list[str]:
    issues: list[str] = []
    policy = getattr(model, "policy", None)
    if policy is None:
        issues.append("model.policy is unavailable")
        return issues

    for name, param in policy.named_parameters():
        tensor = param.detach()
        if bool(th.isfinite(tensor).all()):
            continue
        nan_count = int(th.isnan(tensor).sum().item())
        inf_count = int(th.isinf(tensor).sum().item())
        issues.append(
            f"policy parameter {name} has non-finite values (nan={nan_count}, inf={inf_count})"
        )
    return issues


def _collect_rollout_buffer_issues(model: Any) -> list[str]:
    issues: list[str] = []
    rollout_buffer = getattr(model, "rollout_buffer", None)
    if rollout_buffer is None:
        return issues

    used_size = _rollout_buffer_used_size(model)
    if used_size <= 0:
        return issues

    for field_name in ("actions", "rewards", "returns", "advantages", "values", "log_probs"):
        if not hasattr(rollout_buffer, field_name):
            continue
        array = _slice_rollout_array(getattr(rollout_buffer, field_name), used_size)
        if array is None or array.size <= 0 or not np.issubdtype(array.dtype, np.number):
            continue
        if not bool(np.isfinite(array).all()):
            issues.append(f"rollout_buffer.{field_name} contains non-finite values")

    action_masks = _slice_rollout_array(getattr(rollout_buffer, "action_masks", None), used_size)
    if action_masks is not None and action_masks.size > 0:
        mask_view = np.asarray(action_masks, dtype=bool).reshape(-1, action_masks.shape[-1])
        zero_rows = int(np.sum(mask_view.sum(axis=1) == 0))
        if zero_rows > 0:
            issues.append(
                f"rollout_buffer.action_masks contains {zero_rows} rows with zero valid actions"
            )

    return issues


def _vec_env_wrapper_chain(env: Any) -> list[str]:
    chain: list[str] = []
    current = env
    max_depth = 16
    for _ in range(max_depth):
        if current is None:
            break
        chain.append(type(current).__name__)
        current = getattr(current, "venv", None)
    return chain


def _write_training_diagnostics_report(
    report_path: str | os.PathLike[str],
    *,
    model: Any,
    exception: BaseException | None = None,
    issues: list[str] | None = None,
) -> Path:
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"num_timesteps={int(getattr(model, 'num_timesteps', 0) or 0)}",
        f"n_updates={int(getattr(model, '_n_updates', 0) or 0)}",
        f"policy_class={type(getattr(model, 'policy', None)).__name__}",
    ]

    training_env = model.get_env()
    lines.append(f"training_env_chain={' -> '.join(_vec_env_wrapper_chain(training_env))}")

    rollout_buffer = getattr(model, "rollout_buffer", None)
    used_size = _rollout_buffer_used_size(model)
    if rollout_buffer is not None:
        lines.append(
            "rollout_buffer="
            f"{type(rollout_buffer).__name__}(used={used_size}, "
            f"buffer_size={int(getattr(rollout_buffer, 'buffer_size', 0) or 0)}, "
            f"full={bool(getattr(rollout_buffer, 'full', False))}, "
            f"pos={int(getattr(rollout_buffer, 'pos', 0) or 0)})"
        )

    if issues:
        lines.append("issues:")
        lines.extend(f"- {issue}" for issue in issues)

    if exception is not None:
        lines.append(f"exception_type={type(exception).__name__}")
        lines.append(f"exception_message={exception}")
        lines.append("traceback:")
        lines.extend(traceback.format_exception(type(exception), exception, exception.__traceback__))

    policy = getattr(model, "policy", None)
    if policy is not None:
        lines.append("policy_parameter_diagnostics:")
        param_issues = _collect_policy_parameter_issues(model)
        if param_issues:
            lines.extend(f"- {issue}" for issue in param_issues)
        else:
            lines.append("- all policy parameters are finite")

    if rollout_buffer is not None:
        lines.append("rollout_buffer_diagnostics:")
        for field_name in (
            "observations",
            "actions",
            "rewards",
            "returns",
            "advantages",
            "values",
            "log_probs",
            "action_masks",
        ):
            if not hasattr(rollout_buffer, field_name):
                continue
            lines.extend(
                _format_array_summary(
                    f"rollout_buffer.{field_name}",
                    getattr(rollout_buffer, field_name),
                    used_size=used_size,
                )
            )

    vec_check_nan_env = training_env if isinstance(training_env, VecCheckNan) else None
    if vec_check_nan_env is not None:
        lines.append("vec_check_nan_last_values:")
        lines.extend(
            _format_array_summary("vec_check_nan.last_actions", getattr(vec_check_nan_env, "_actions", None))
        )
        lines.extend(
            _format_array_summary(
                "vec_check_nan.last_observations",
                getattr(vec_check_nan_env, "_observations", None),
            )
        )

    target.write_text("\n".join(lines), encoding="utf-8")
    return target


class CampaignNumericsDiagnosticsCallback(BaseCallback):

    def __init__(self, report_dir: str, experiment: Any | None = None, verbose: int = 1):
        super().__init__(verbose)
        self.report_dir = Path(report_dir)
        self.experiment = experiment
        self._zero_mask_warning_reported = False

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        issues = []
        policy_issues = _collect_policy_parameter_issues(self.model)
        rollout_issues = _collect_rollout_buffer_issues(self.model)
        issues.extend(policy_issues)
        issues.extend(rollout_issues)
        if not issues:
            return

        severe_issues = [issue for issue in issues if "non-finite" in issue]
        zero_mask_issues = [issue for issue in issues if "zero valid actions" in issue]

        should_write_warning = bool(zero_mask_issues) and not self._zero_mask_warning_reported
        if not severe_issues and not should_write_warning:
            return

        report_kind = "rollout_error" if severe_issues else "rollout_warning"
        report_path = self.report_dir / f"{report_kind}_step_{self.num_timesteps}.txt"
        written_path = _write_training_diagnostics_report(
            report_path,
            model=self.model,
            issues=issues,
        )
        if self.verbose:
            print(f"[DIAG] Wrote diagnostics to {written_path}")
            for issue in issues:
                print(f"[DIAG] {issue}")
        if self.experiment is not None:
            self.experiment.log_other("diagnostics_last_report", str(written_path))

        if zero_mask_issues:
            self._zero_mask_warning_reported = True

        if severe_issues:
            raise RuntimeError(
                f"Numerics diagnostics found a non-finite training state. See {written_path}"
            )


class CampaignCheckpointCallback(BaseCallback):

    def __init__(
        self,
        save_dir: str,
        milestone_steps: int = 500_000,
        start_timesteps: int = 0,
        experiment: Any | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.milestone_steps = int(milestone_steps)
        self.experiment = experiment
        self.next_milestone = self.milestone_steps
        while self.next_milestone <= int(start_timesteps):
            self.next_milestone += self.milestone_steps
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


class WallClockStopCallback(BaseCallback):

    def __init__(self, wall_time_seconds: float, verbose: int = 1):
        super().__init__(verbose)
        self.wall_time_seconds = float(wall_time_seconds)
        self.started_at: float | None = None
        self.elapsed_seconds = 0.0
        self.stop_requested = False

    def _on_training_start(self) -> None:
        self.started_at = time.monotonic()

    def _on_step(self) -> bool:
        if self.started_at is None:
            self.started_at = time.monotonic()
        self.elapsed_seconds = time.monotonic() - self.started_at
        if self.elapsed_seconds < self.wall_time_seconds:
            return True
        if not self.stop_requested and self.verbose:
            print(
                "[WallClock] Training limit reached after "
                f"{self.elapsed_seconds:.1f}s at {self.num_timesteps:,} steps; "
                "saving the final model."
            )
        self.stop_requested = True
        return False

    def _on_training_end(self) -> None:
        if self.started_at is not None:
            self.elapsed_seconds = time.monotonic() - self.started_at


class SaveVecNormalizeOnBestCallback(BaseCallback):

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

    def __init__(
        self,
        *args,
        use_masking: bool = True,
        stochastic_eval_episodes: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_masking = bool(use_masking)
        self.stochastic_eval_episodes = max(0, int(stochastic_eval_episodes))
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

    def _evaluate_policy(self, *, deterministic: bool, n_eval_episodes: int):
        if self.use_masking:
            if maskable_evaluate_policy is None:
                raise RuntimeError("Maskable evaluation is unavailable, but use_masking=True.")
            return maskable_evaluate_policy(
                self.model,  # type: ignore[arg-type]
                self.eval_env,
                n_eval_episodes=n_eval_episodes,
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
            n_eval_episodes=n_eval_episodes,
            render=self.render,
            deterministic=deterministic,
            return_episode_rewards=True,
            warn=self.warn,
            callback=self._log_success_callback,
        )

    def _run_campaign_eval(
        self,
        *,
        deterministic: bool,
        n_eval_episodes: int,
    ) -> tuple[list[float], list[int], list[bool], list[bool]]:
        self._is_success_buffer = []
        self._eval_victory_buffer = []
        episode_rewards, episode_lengths = self._evaluate_policy(
            deterministic=deterministic,
            n_eval_episodes=n_eval_episodes,
        )
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
                deterministic=True,
                n_eval_episodes=self.n_eval_episodes,
            )
            stoch_rewards: list[float] = []
            stoch_lengths: list[int] = []
            stoch_successes: list[bool] = []
            stoch_victories: list[bool] = []
            if self.stochastic_eval_episodes > 0:
                stoch_rewards, stoch_lengths, stoch_successes, stoch_victories = (
                    self._run_campaign_eval(
                        deterministic=False,
                        n_eval_episodes=self.stochastic_eval_episodes,
                    )
                )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(det_rewards)
                self.evaluations_length.append(det_lengths)
                self.evaluations_victories.append(det_victories)

                kwargs = {
                    "victories": self.evaluations_victories,
                    "deterministic_results": self.evaluations_results,
                    "deterministic_ep_lengths": self.evaluations_length,
                    "deterministic_victories": self.evaluations_victories,
                }
                if det_successes:
                    self.evaluations_successes.append(det_successes)
                    kwargs["successes"] = self.evaluations_successes
                    kwargs["deterministic_successes"] = self.evaluations_successes
                if self.stochastic_eval_episodes > 0:
                    self.stochastic_evaluations_results.append(stoch_rewards)
                    self.stochastic_evaluations_length.append(stoch_lengths)
                    self.stochastic_evaluations_victories.append(stoch_victories)
                    kwargs["stochastic_results"] = self.stochastic_evaluations_results
                    kwargs["stochastic_ep_lengths"] = self.stochastic_evaluations_length
                    kwargs["stochastic_victories"] = self.stochastic_evaluations_victories
                if self.stochastic_eval_episodes > 0 and stoch_successes:
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
            stochastic_enabled = self.stochastic_eval_episodes > 0
            stochastic_mean_reward = float(np.mean(stoch_rewards)) if stochastic_enabled else 0.0
            stochastic_std_reward = float(np.std(stoch_rewards)) if stochastic_enabled else 0.0
            stochastic_mean_ep_length = float(np.mean(stoch_lengths)) if stochastic_enabled else 0.0
            stochastic_std_ep_length = float(np.std(stoch_lengths)) if stochastic_enabled else 0.0
            stochastic_victory_rate = float(np.mean(stoch_victories)) if stoch_victories else 0.0
            self.last_mean_reward = mean_reward
            self.last_victory_rate = victory_rate
            self.last_stochastic_victory_rate = stochastic_victory_rate

            if self.verbose >= 1:
                if stochastic_enabled:
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
                else:
                    print(
                        f"Eval num_timesteps={self.num_timesteps}, "
                        f"det_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                    )
                    print(f"Episode length: det={mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                    print(f"Victory rate: det={100 * victory_rate:.2f}%")

            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/victory_rate", victory_rate)
            self.logger.record("eval/deterministic_mean_reward", mean_reward)
            self.logger.record("eval/deterministic_mean_ep_length", mean_ep_length)
            self.logger.record("eval/deterministic_victory_rate", victory_rate)
            if stochastic_enabled:
                self.logger.record("eval/stochastic_mean_reward", stochastic_mean_reward)
                self.logger.record("eval/stochastic_mean_ep_length", stochastic_mean_ep_length)
                self.logger.record("eval/stochastic_victory_rate", stochastic_victory_rate)

            if det_successes:
                success_rate = float(np.mean(det_successes))
                if self.verbose >= 1:
                    print(f"Success rate: det={100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)
                self.logger.record("eval/deterministic_success_rate", success_rate)
            if stochastic_enabled and stoch_successes:
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

    def __init__(
        self,
        log_freq: int = 1000,
        episode_window: int = 100,
        experiment: Any | None = None,
        report_path: str | os.PathLike[str] | None = None,
        story_path: str | os.PathLike[str] | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_freq = max(1, int(log_freq))
        self.episode_window = max(1, int(episode_window))
        self.step_window = max(16, min(self.log_freq, 256))
        self.experiment = experiment
        self.report_path = Path(report_path) if report_path is not None else None
        if self.report_path is not None:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.story_path = Path(story_path) if story_path is not None else None
        if self.story_path is not None:
            self.story_path.parent.mkdir(parents=True, exist_ok=True)

        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.recent_episode_rewards = deque(maxlen=self.episode_window)
        self.recent_episode_lengths = deque(maxlen=self.episode_window)
        self.recent_blue_hp_ratios = deque(maxlen=self.episode_window)
        self.recent_unit_upgrades = deque(maxlen=self.episode_window)
        self.recent_episode_max_tier = deque(maxlen=self.episode_window)
        self.recent_enemy_defeat_rewards = deque(maxlen=self.episode_window)
        self.recent_blue_exp_rewards = deque(maxlen=self.episode_window)
        self.recent_final_objective_rewards = deque(maxlen=self.episode_window)
        self.recent_castle_heal_uses = deque(maxlen=self.episode_window)
        self.recent_castle_heal_episode_flags = deque(maxlen=self.episode_window)
        self.recent_castle_healed_hp = deque(maxlen=self.episode_window)
        self.episode_chests_collected: list[float] = []
        self.recent_chests_collected = deque(maxlen=self.episode_window)
        self.recent_chest_episode_flags = deque(maxlen=self.episode_window)
        self.episode_buildings_built: list[float] = []
        self.recent_buildings_built = deque(maxlen=self.episode_window)
        self.recent_building_episode_flags = deque(maxlen=self.episode_window)
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
        self.episode_summoned_units: list[float] = []
        self.recent_summoned_units = deque(maxlen=self.episode_window)
        self.recent_summon_episode_flags = deque(maxlen=self.episode_window)
        self.episode_hired_units: list[float] = []
        self.recent_hired_units = deque(maxlen=self.episode_window)
        self.recent_hire_episode_flags = deque(maxlen=self.episode_window)
        self.episode_battle_items_equipped: list[float] = []
        self.recent_battle_items_equipped = deque(maxlen=self.episode_window)
        self.recent_battle_item_equip_episode_flags = deque(maxlen=self.episode_window)
        self.episode_battle_items_used: list[float] = []
        self.recent_battle_items_used = deque(maxlen=self.episode_window)
        self.recent_battle_item_use_episode_flags = deque(maxlen=self.episode_window)
        self.episode_unit_swaps: list[float] = []
        self.recent_unit_swaps = deque(maxlen=self.episode_window)
        self.recent_unit_swap_episode_flags = deque(maxlen=self.episode_window)
        self.episode_scripted_bot_victories: list[float] = []
        self.recent_scripted_bot_victories = deque(maxlen=self.episode_window)
        self.recent_scripted_bot_victory_episode_flags = deque(maxlen=self.episode_window)
        self.recent_turns = deque(maxlen=self.step_window)
        self.recent_gold = deque(maxlen=self.step_window)
        self.recent_moves = deque(maxlen=self.step_window)

        self.result_counter: Counter[str] = Counter()
        self.window_result_counter: Counter[str] = Counter()
        self.battle_counter: Counter[str] = Counter()
        self.window_battle_counter: Counter[str] = Counter()
        self.victory_reason_counter: Counter[str] = Counter()
        self.window_victory_reason_counter: Counter[str] = Counter()
        self.enemy_encounter_counter: Counter[str] = Counter()
        self.enemy_victory_counter: Counter[str] = Counter()
        self.window_enemy_encounter_counter: Counter[str] = Counter()
        self.window_enemy_victory_counter: Counter[str] = Counter()

        self.victories = 0
        self.defeats = 0
        self.timeouts = 0
        self.total_unit_upgrades = 0
        self.max_unit_tier_reached = 0
        self.total_castle_heal_uses = 0
        self.total_castle_healed_hp = 0.0
        self.castle_heal_episodes = 0
        self.total_chests_collected = 0
        self.chest_collection_episodes = 0
        self.total_buildings_built = 0
        self.building_episodes = 0
        self.total_ruins_cleared = 0
        self.ruin_clear_episodes = 0
        self.total_sold_items = 0
        self.total_merchant_sale_gold = 0.0
        self.sale_episodes = 0
        self.total_spells_learned = 0
        self.spell_learning_episodes = 0
        self.total_spell_casts = 0
        self.spell_cast_episodes = 0
        self.total_summoned_units = 0
        self.summon_episodes = 0
        self.total_hired_units = 0
        self.hire_episodes = 0
        self.total_battle_items_equipped = 0
        self.battle_item_equip_episodes = 0
        self.total_battle_items_used = 0
        self.battle_item_use_episodes = 0
        self.total_unit_swaps = 0
        self.unit_swap_episodes = 0
        self.total_scripted_bot_victories = 0
        self.scripted_bot_victory_episodes = 0
        self.best_recent_victory_rate = 0.0
        self._last_logged_step = 0
        self._episode_castle_heal_uses: list[int] = []
        self._episode_castle_heal_flags: list[bool] = []
        self._episode_castle_healed_hp: list[float] = []
        self._episode_chests_collected: list[int] = []
        self._episode_buildings_built: list[int] = []
        self._episode_ruins_cleared: list[int] = []
        self._episode_sold_items: list[int] = []
        self._episode_sale_gold: list[float] = []
        self._episode_spells_learned: list[int] = []
        self._episode_spell_casts: list[int] = []
        self._episode_summoned_units: list[int] = []
        self._episode_hired_units: list[int] = []
        self._episode_battle_items_equipped: list[int] = []
        self._episode_battle_items_used: list[int] = []
        self._episode_unit_swaps: list[int] = []
        self._episode_max_tier: list[int] = []
        self._episode_unit_upgrades: list[int] = []

    def _ensure_env_trackers(self, env_count: int) -> None:
        if env_count <= len(self._episode_castle_heal_uses):
            return
        missing = env_count - len(self._episode_castle_heal_uses)
        self._episode_castle_heal_uses.extend([0] * missing)
        self._episode_castle_heal_flags.extend([False] * missing)
        self._episode_castle_healed_hp.extend([0.0] * missing)
        self._episode_chests_collected.extend([0] * missing)
        self._episode_buildings_built.extend([0] * missing)
        self._episode_ruins_cleared.extend([0] * missing)
        self._episode_sold_items.extend([0] * missing)
        self._episode_sale_gold.extend([0.0] * missing)
        self._episode_spells_learned.extend([0] * missing)
        self._episode_spell_casts.extend([0] * missing)
        self._episode_summoned_units.extend([0] * missing)
        self._episode_hired_units.extend([0] * missing)
        self._episode_battle_items_equipped.extend([0] * missing)
        self._episode_battle_items_used.extend([0] * missing)
        self._episode_unit_swaps.extend([0] * missing)
        self._episode_max_tier.extend([0] * missing)
        self._episode_unit_upgrades.extend([0] * missing)

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
        if isinstance(collected, (list, tuple)):
            chest_count = len(collected)
        else:
            try:
                chest_count = int(info.get("collected_chests_count", 0) or 0)
            except (TypeError, ValueError):
                chest_count = 0
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


        self._episode_chests_collected[env_index] = 0

    def _consume_buildings(self, info: dict[str, Any], env_index: int) -> None:
        if not (bool(info.get("build_action")) and bool(info.get("built"))):
            return
        self.total_buildings_built += 1
        self._episode_buildings_built[env_index] += 1

    def _finalize_episode_buildings(self, env_index: int) -> None:
        building_count = float(self._episode_buildings_built[env_index])
        self.episode_buildings_built.append(building_count)
        self.recent_buildings_built.append(building_count)
        self.recent_building_episode_flags.append(1.0 if building_count > 0.0 else 0.0)
        if building_count > 0.0:
            self.building_episodes += 1

        self._episode_buildings_built[env_index] = 0

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


        self._episode_spell_casts[env_index] = 0

    def _consume_summons(self, info: dict[str, Any], env_index: int) -> None:
        if str(info.get("spell_kind", "") or "") != "summon_battle":
            return
        if not info.get("spell_cast_executed"):
            return
        try:
            summoned_units = int(info.get("spell_units_affected", 1) or 1)
        except (TypeError, ValueError):
            summoned_units = 1
        summoned_units = max(1, summoned_units)
        self.total_summoned_units += summoned_units
        self._episode_summoned_units[env_index] += summoned_units

    def _finalize_episode_summons(self, env_index: int) -> None:
        summoned_units = float(self._episode_summoned_units[env_index])
        summon_flag = summoned_units > 0.0

        self.episode_summoned_units.append(summoned_units)
        self.recent_summoned_units.append(summoned_units)
        self.recent_summon_episode_flags.append(1.0 if summon_flag else 0.0)
        if summon_flag:
            self.summon_episodes += 1


        self._episode_summoned_units[env_index] = 0

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


        self._episode_hired_units[env_index] = 0

    def _consume_battle_item_activity(self, info: dict[str, Any], env_index: int) -> None:
        equipped = bool(info.get("equip_battle_item_action")) and bool(
            info.get("equip_battle_item_applied")
        )
        used = (
            bool(info.get("battle_hero_item_action"))
            and bool(info.get("battle_hero_item_applied"))
            and (
                bool(info.get("battle_hero_item_charge_spent"))
                or bool(info.get("battle_hero_item_consumed"))
            )
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


        self._episode_battle_items_equipped[env_index] = 0
        self._episode_battle_items_used[env_index] = 0

    def _consume_unit_upgrades(self, info: dict[str, Any], env_index: int) -> None:
        if info.get("battle_result") != "victory":
            return
        unit_upgrades = info.get("unit_upgrades")
        if unit_upgrades is None:
            return
        try:
            cnt = int(unit_upgrades)
        except (TypeError, ValueError):
            return
        self.total_unit_upgrades += cnt
        self._episode_unit_upgrades[env_index] += cnt

    def _finalize_episode_unit_upgrades(self, env_index: int) -> None:
        self.recent_unit_upgrades.append(float(self._episode_unit_upgrades[env_index]))
        self._episode_unit_upgrades[env_index] = 0

    def _consume_unit_tier(self, info: dict[str, Any], env_index: int) -> None:
        max_tier = info.get("unit_upgrade_max_tier")
        if max_tier is None:
            return
        try:
            tier = int(max_tier)
        except (TypeError, ValueError):
            return
        self._episode_max_tier[env_index] = max(self._episode_max_tier[env_index], tier)

    def _finalize_episode_unit_tier(self, env_index: int) -> None:
        self.recent_episode_max_tier.append(float(self._episode_max_tier[env_index]))
        self._episode_max_tier[env_index] = 0

    def _consume_unit_swaps(self, info: dict[str, Any], env_index: int) -> None:
        if not (bool(info.get("unit_swap_action")) and bool(info.get("unit_swap_applied"))):
            return
        self.total_unit_swaps += 1
        self._episode_unit_swaps[env_index] += 1

    def _finalize_episode_unit_swaps(self, env_index: int) -> None:
        swap_count = float(self._episode_unit_swaps[env_index])
        swap_flag = swap_count > 0.0

        self.episode_unit_swaps.append(swap_count)
        self.recent_unit_swaps.append(swap_count)
        self.recent_unit_swap_episode_flags.append(1.0 if swap_flag else 0.0)
        if swap_flag:
            self.unit_swap_episodes += 1

        self._episode_unit_swaps[env_index] = 0

    def _consume_info(self, info: dict[str, Any], env_index: int | None = None) -> None:
        if env_index is not None:
            self._consume_castle_heal(info, env_index)
            self._consume_chests(info, env_index)
            self._consume_buildings(info, env_index)
            self._consume_ruins(info, env_index)
            self._consume_merchant_sales(info, env_index)
            self._consume_spell_learning(info, env_index)
            self._consume_spell_casts(info, env_index)
            self._consume_summons(info, env_index)
            self._consume_hires(info, env_index)
            self._consume_battle_item_activity(info, env_index)
            self._consume_unit_swaps(info, env_index)
            self._consume_unit_upgrades(info, env_index)
            self._consume_unit_tier(info, env_index)

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
                self._finalize_episode_buildings(env_index)
                self._finalize_episode_ruins(env_index)
                self._finalize_episode_sales(env_index)
                self._finalize_episode_spell_learning(env_index)
                self._finalize_episode_spell_casts(env_index)
                self._finalize_episode_summons(env_index)
                self._finalize_episode_hires(env_index)
                self._finalize_episode_battle_item_activity(env_index)
                self._finalize_episode_unit_swaps(env_index)
                self._finalize_episode_unit_upgrades(env_index)
                self._finalize_episode_unit_tier(env_index)
                self._finalize_episode_scripted_bot(info)

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

        raw_enemy_id = (
            info.get("enemy_id")
            if info.get("enemy_id") is not None
            else info.get("target_enemy_id")
        )
        try:
            enemy_key = f"enemy_{int(raw_enemy_id)}"
        except (TypeError, ValueError):
            enemy_key = ""
        if enemy_key and (
            bool(info.get("battle_triggered", False))
            or bool(info.get("spell_enemy_defeated", False))
            or battle_result
        ):
            self.enemy_encounter_counter[enemy_key] += 1
            self.window_enemy_encounter_counter[enemy_key] += 1
        if enemy_key and (
            battle_result == "victory" or bool(info.get("spell_enemy_defeated", False))
        ):
            self.enemy_victory_counter[enemy_key] += 1
            self.window_enemy_victory_counter[enemy_key] += 1

        victory_reason = info.get("campaign_victory_reason")
        if victory_reason:
            reason_key = str(victory_reason)
            self.victory_reason_counter[reason_key] += 1
            self.window_victory_reason_counter[reason_key] += 1

        final_objective_reward = info.get("final_objective_reward")
        if final_objective_reward is not None:
            try:
                self.recent_final_objective_rewards.append(float(final_objective_reward))
            except (TypeError, ValueError):
                pass

        max_tier = info.get("unit_upgrade_max_tier")
        if max_tier is not None:
            try:
                self.max_unit_tier_reached = max(
                    int(self.max_unit_tier_reached), int(max_tier)
                )
            except (TypeError, ValueError):
                pass

    def _finalize_episode_scripted_bot(self, info: dict[str, Any]) -> None:
        try:
            victories = int(info.get("scripted_capital_bot_enemies_defeated", 0) or 0)
        except (TypeError, ValueError):
            victories = 0

        victories = max(0, victories)
        self.total_scripted_bot_victories += victories
        self.episode_scripted_bot_victories.append(float(victories))
        self.recent_scripted_bot_victories.append(float(victories))
        self.recent_scripted_bot_victory_episode_flags.append(1.0 if victories > 0 else 0.0)
        if victories > 0:
            self.scripted_bot_victory_episodes += 1

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
            "campaign/victory_rate": _safe_rate(self.victories, total_episodes),
            "campaign/defeat_rate": _safe_rate(self.defeats, total_episodes),
            "campaign/timeout_rate": _safe_rate(self.timeouts, total_episodes),
            "campaign/battle_victories_per_episode_mean": _safe_rate(
                self.battle_counter.get("victory", 0),
                total_episodes,
            ),
            "campaign/battle_win_rate": _safe_rate(
                self.battle_counter.get("victory", 0),
                total_battles,
            ),
            "campaign/unit_upgrades_total": float(self.total_unit_upgrades),
            "campaign/castle_heal_episodes_rate": _safe_rate(
                self.castle_heal_episodes,
                total_episodes,
            ),
            "campaign/chests_collected_per_episode_mean": _safe_rate(
                self.total_chests_collected,
                total_episodes,
            ),
            "campaign/buildings_built_per_episode_mean": _safe_rate(
                self.total_buildings_built,
                total_episodes,
            ),
            "campaign/building_episodes_rate": _safe_rate(
                self.building_episodes,
                total_episodes,
            ),
            "campaign/ruins_cleared_per_episode_mean": _safe_rate(
                self.total_ruins_cleared,
                total_episodes,
            ),
            "campaign/ruin_clear_episodes_rate": _safe_rate(
                self.ruin_clear_episodes,
                total_episodes,
            ),
            "campaign/sold_items_per_episode_mean": _safe_rate(
                self.total_sold_items,
                total_episodes,
            ),
            "campaign/merchant_sale_gold_per_episode_mean": _safe_rate(
                self.total_merchant_sale_gold,
                total_episodes,
            ),
            "campaign/spells_learned_per_episode_mean": _safe_rate(
                self.total_spells_learned,
                total_episodes,
            ),
            "campaign/spell_casts_per_episode_mean": _safe_rate(
                self.total_spell_casts,
                total_episodes,
            ),
            "campaign/spell_cast_episodes_rate": _safe_rate(
                self.spell_cast_episodes,
                total_episodes,
            ),
            "campaign/summoned_units_per_episode_mean": _safe_rate(
                self.total_summoned_units,
                total_episodes,
            ),
            "campaign/summon_episodes_rate": _safe_rate(
                self.summon_episodes,
                total_episodes,
            ),
            "campaign/hired_units_per_episode_mean": _safe_rate(
                self.total_hired_units,
                total_episodes,
            ),
            "campaign/hire_episodes_rate": _safe_rate(
                self.hire_episodes,
                total_episodes,
            ),
            "campaign/battle_items_used_per_episode_mean": _safe_rate(
                self.total_battle_items_used,
                total_episodes,
            ),
            "campaign/unit_swaps_per_episode_mean": _safe_rate(
                self.total_unit_swaps,
                total_episodes,
            ),
            "campaign/unit_swap_episodes_rate": _safe_rate(
                self.unit_swap_episodes,
                total_episodes,
            ),
            "campaign/scripted_bot_victories_per_episode_mean": _safe_rate(
                self.total_scripted_bot_victories,
                total_episodes,
            ),
            "campaign/scripted_bot_victory_episodes_rate": _safe_rate(
                self.scripted_bot_victory_episodes,
                total_episodes,
            ),
            "campaign/battle_item_use_episodes_rate": _safe_rate(
                self.battle_item_use_episodes,
                total_episodes,
            ),
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
            "campaign/window/battle_win_rate": _safe_rate(
                self.window_battle_counter.get("victory", 0),
                window_battles,
            ),
            "campaign/window/chests_collected_mean": _safe_mean(self.recent_chests_collected),
            "campaign/window/buildings_built_mean": _safe_mean(self.recent_buildings_built),
            "campaign/window/ruins_cleared_mean": _safe_mean(self.recent_ruins_cleared),
            "campaign/window/spell_casts_mean": _safe_mean(self.recent_spell_casts),
            "campaign/window/hired_units_mean": _safe_mean(self.recent_hired_units),
            "campaign/window/unit_swaps_mean": _safe_mean(self.recent_unit_swaps),
            "campaign/window/scripted_bot_victories_mean": _safe_mean(
                self.recent_scripted_bot_victories
            ),
            "campaign/best_recent_victory_rate": self.best_recent_victory_rate,
        }

        return payload

    @staticmethod
    def _counter_top(counter: Counter[str], limit: int = 12) -> dict[str, int]:
        return {str(key): int(value) for key, value in counter.most_common(int(limit))}

    def _build_behavior_report(self, payload: dict[str, float]) -> dict[str, Any]:
        total_episodes = int(sum(self.result_counter.values()))
        window_episodes = int(sum(self.window_result_counter.values()))
        dragon_key = "enemy_31"
        recent_activity = {
            "magic_spell_casts_mean": float(payload.get("campaign/window/spell_casts_mean", 0.0)),
            "summons_mean": _safe_mean(self.recent_summoned_units),
            "hires_mean": float(payload.get("campaign/window/hired_units_mean", 0.0)),
            "unit_swaps_mean": float(payload.get("campaign/window/unit_swaps_mean", 0.0)),
            "chests_mean": float(payload.get("campaign/window/chests_collected_mean", 0.0)),
            "buildings_built_mean": float(payload.get("campaign/window/buildings_built_mean", 0.0)),
            "ruins_mean": float(payload.get("campaign/window/ruins_cleared_mean", 0.0)),
            "battle_items_used_mean": _safe_mean(self.recent_battle_items_used),
            "battle_items_equipped_mean": _safe_mean(self.recent_battle_items_equipped),
            "castle_heal_uses_mean": _safe_mean(self.recent_castle_heal_uses),
            "castle_healed_hp_mean": _safe_mean(self.recent_castle_healed_hp),
            "sold_items_mean": _safe_mean(self.recent_sold_items),
            "merchant_sale_gold_mean": _safe_mean(self.recent_sale_gold),
            "unit_upgrades_mean": _safe_mean(self.recent_unit_upgrades),
            "max_unit_tier_reached": int(self.max_unit_tier_reached),
            "episode_max_tier_mean": _safe_mean(self.recent_episode_max_tier),
            "tier3_episode_fraction": (
                _safe_mean([1.0 if t >= 3 else 0.0 for t in self.recent_episode_max_tier])
                if self.recent_episode_max_tier
                else 0.0
            ),
            "tier4_episode_fraction": (
                _safe_mean([1.0 if t >= 4 else 0.0 for t in self.recent_episode_max_tier])
                if self.recent_episode_max_tier
                else 0.0
            ),
        }
        uses = {
            "magic": recent_activity["magic_spell_casts_mean"] > 0.0,
            "summon_magic": recent_activity["summons_mean"] > 0.0,
            "hire": recent_activity["hires_mean"] > 0.0,
            "battle_items_or_potions": (
                recent_activity["battle_items_used_mean"] > 0.0
                or recent_activity["battle_items_equipped_mean"] > 0.0
            ),
            "castle_heal": recent_activity["castle_heal_uses_mean"] > 0.0,
            "merchant_sales": recent_activity["sold_items_mean"] > 0.0,
            "unit_upgrades": recent_activity["unit_upgrades_mean"] > 0.0,
            "ruins": recent_activity["ruins_mean"] > 0.0,
            "chests": recent_activity["chests_mean"] > 0.0,
            "buildings": recent_activity["buildings_built_mean"] > 0.0,
            "unit_swaps": recent_activity["unit_swaps_mean"] > 0.0,
        }
        return {
            "step": int(self.num_timesteps),
            "episodes_total": total_episodes,
            "episodes_window": window_episodes,
            "reward_mean_window": _safe_mean(self.recent_episode_rewards),
            "episode_length_mean_window": _safe_mean(self.recent_episode_lengths),
            "metrics": payload,
            "uses": uses,
            "recent_activity": recent_activity,
            "results_total": {str(k): int(v) for k, v in self.result_counter.items()},
            "results_window": {str(k): int(v) for k, v in self.window_result_counter.items()},
            "victory_reasons_total": {
                str(k): int(v) for k, v in self.victory_reason_counter.items()
            },
            "victory_reasons_window": {
                str(k): int(v) for k, v in self.window_victory_reason_counter.items()
            },
            "enemy_encounters_total_top": self._counter_top(self.enemy_encounter_counter),
            "enemy_victories_total_top": self._counter_top(self.enemy_victory_counter),
            "enemy_encounters_window_top": self._counter_top(self.window_enemy_encounter_counter),
            "enemy_victories_window_top": self._counter_top(self.window_enemy_victory_counter),
            "dragon_progress": {
                "encounters_total": int(self.enemy_encounter_counter.get(dragon_key, 0)),
                "victories_total": int(self.enemy_victory_counter.get(dragon_key, 0)),
                "encounters_window": int(self.window_enemy_encounter_counter.get(dragon_key, 0)),
                "victories_window": int(self.window_enemy_victory_counter.get(dragon_key, 0)),
                "objective_victories_total": int(
                    self.victory_reason_counter.get("green_dragon_defeated", 0)
                ),
                "objective_victories_window": int(
                    self.window_victory_reason_counter.get("green_dragon_defeated", 0)
                ),
                "final_objective_reward_mean_window": _safe_mean(
                    self.recent_final_objective_rewards
                ),
            },
        }

    def _write_behavior_report(self, payload: dict[str, float]) -> None:
        report = self._build_behavior_report(payload)
        if self.report_path is not None:
            with self.report_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
        if self.story_path is not None:
            with self.story_path.open("a", encoding="utf-8") as handle:
                handle.write(self._format_behavior_story(report) + "\n\n")

    @staticmethod
    def _format_counter(counter: dict[str, int]) -> str:
        if not counter:
            return "пока никого заметно не выделяется"
        return ", ".join(f"{key}={value}" for key, value in counter.items())

    @staticmethod
    def _yes_no(value: object) -> str:
        return "да" if bool(value) else "нет"

    def _format_behavior_story(self, report: dict[str, Any]) -> str:
        metrics = dict(report.get("metrics", {}) or {})
        uses = dict(report.get("uses", {}) or {})
        activity = dict(report.get("recent_activity", {}) or {})
        dragon = dict(report.get("dragon_progress", {}) or {})
        step = int(report.get("step", 0) or 0)
        episodes_window = int(report.get("episodes_window", 0) or 0)
        reward_mean = float(report.get("reward_mean_window", 0.0) or 0.0)
        length_mean = float(report.get("episode_length_mean_window", 0.0) or 0.0)
        battle_win_rate = float(metrics.get("campaign/window/battle_win_rate", 0.0) or 0.0)
        victory_rate = float(metrics.get("campaign/window/victory_rate", 0.0) or 0.0)
        timeout_rate = float(metrics.get("campaign/window/timeout_rate", 0.0) or 0.0)

        if dragon.get("victories_window", 0):
            dragon_story = "В этом окне агент уже побеждал зеленого дракона."
        elif dragon.get("encounters_window", 0):
            dragon_story = "До дракона агент добирался, но победы над ним в этом окне еще нет."
        elif report.get("enemy_victories_window_top"):
            dragon_story = "До дракона в этом окне не добрался, но путь через карту уже дает боевые победы."
        else:
            dragon_story = "До дракона пока не добирается; поведение еще похоже на раннюю разведку/набор промежуточных наград."

        mechanics = (
            f"магия: {self._yes_no(uses.get('magic'))}, "
            f"призывы: {self._yes_no(uses.get('summon_magic'))}, "
            f"строительство: {self._yes_no(uses.get('buildings'))}, "
            f"найм: {self._yes_no(uses.get('hire'))}, "
            f"боевые предметы/зелья: {self._yes_no(uses.get('battle_items_or_potions'))}, "
            f"лечение в замке/храме: {self._yes_no(uses.get('castle_heal'))}, "
            f"продажи у торговца: {self._yes_no(uses.get('merchant_sales'))}, "
            f"перестановки юнитов: {self._yes_no(uses.get('unit_swaps'))}"
        )

        return "\n".join(
            [
                f"## Оценка поведения на шаге {step:,}",
                "",
                (
                    f"За последнее окно завершилось {episodes_window} эпизодов. "
                    f"Средняя награда около {reward_mean:.2f}, средняя длина эпизода {length_mean:.1f} шага. "
                    f"Оконный winrate кампании {100.0 * victory_rate:.1f}%, "
                    f"battle winrate {100.0 * battle_win_rate:.1f}%, "
                    f"timeout rate {100.0 * timeout_rate:.1f}%."
                ),
                "",
                f"По механикам: {mechanics}.",
                (
                    "Средние действия на эпизод в окне: "
                    f"заклинания {float(activity.get('magic_spell_casts_mean', 0.0) or 0.0):.2f}, "
                    f"постройки {float(activity.get('buildings_built_mean', 0.0) or 0.0):.2f}, "
                    f"найм {float(activity.get('hires_mean', 0.0) or 0.0):.2f}, "
                    f"боевые предметы использованы {float(activity.get('battle_items_used_mean', 0.0) or 0.0):.2f}, "
                    f"лечение {float(activity.get('castle_heal_uses_mean', 0.0) or 0.0):.2f}, "
                    f"апгрейды юнитов {float(activity.get('unit_upgrades_mean', 0.0) or 0.0):.2f}. "
                    f"Макс. тир прокачки за прогон: {int(activity.get('max_unit_tier_reached', 0) or 0)}; "
                    f"средний макс. тир за эпизод в окне: {float(activity.get('episode_max_tier_mean', 0.0) or 0.0):.2f}; "
                    f"доля эпизодов с тиром >=3: {100.0 * float(activity.get('tier3_episode_fraction', 0.0) or 0.0):.1f}%; "
                    f"с тиром >=4: {100.0 * float(activity.get('tier4_episode_fraction', 0.0) or 0.0):.1f}%."
                ),
                "",
                (
                    "Встреченные враги в окне: "
                    f"{self._format_counter(dict(report.get('enemy_encounters_window_top', {}) or {}))}. "
                    "Победы в окне: "
                    f"{self._format_counter(dict(report.get('enemy_victories_window_top', {}) or {}))}."
                ),
                (
                    f"Прогресс к дракону: встреч с enemy_31 всего {int(dragon.get('encounters_total', 0) or 0)}, "
                    f"в текущем окне {int(dragon.get('encounters_window', 0) or 0)}; "
                    f"побед над ним всего {int(dragon.get('victories_total', 0) or 0)}, "
                    f"в текущем окне {int(dragon.get('victories_window', 0) or 0)}. {dragon_story}"
                ),
                "",
                (
                    "Итог по ощущению траектории: "
                    "если растут battle winrate, число побед над дальними enemy id и используются найм/магия/лечение, "
                    "агент собирает инструменты для похода на дракона; если в окне только короткие эпизоды и нет встреч с сильными врагами, "
                    "он пока застрял на ранней экономике или локальных боях."
                ),
            ]
        )

    def _reset_log_window(self) -> None:
        self.window_result_counter.clear()
        self.window_battle_counter.clear()
        self.window_victory_reason_counter.clear()
        self.window_enemy_encounter_counter.clear()
        self.window_enemy_victory_counter.clear()

    def _log_to_experiment(self, *, force: bool = False) -> None:
        if not force and self.num_timesteps - self._last_logged_step < self.log_freq:
            return

        payload = self._build_log_payload()
        self._write_behavior_report(payload)
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
        self._ensure_env_trackers(len(infos))
        for env_index, info in enumerate(infos):
            self._consume_info(info, env_index=env_index)
        self._log_to_experiment()
        return True

    def _on_training_end(self) -> None:
        self._log_to_experiment(force=True)

    @staticmethod
    def _normalize_plot_path(path: str | os.PathLike[str]) -> Path:
        raw_path = os.fspath(path)
        if isinstance(raw_path, str):
            raw_path = "".join(ch for ch in raw_path.strip().strip('"') if ch >= " " and ch != "\x7f")
        return Path(raw_path)

    @classmethod
    def _save_plot_figure(cls, fig, path: str | os.PathLike[str]) -> Path | None:
        import matplotlib.pyplot as plt

        target = cls._normalize_plot_path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(target), dpi=160)
            return target
        except OSError as exc:
            print(f"[WARN] Failed to save campaign plot to {target}: {exc}")
            return None
        finally:
            plt.close(fig)

    def save_spell_cast_activity_plot(self, path: str | os.PathLike[str]) -> Path | None:
        if not self.episode_spell_casts:
            return None

        import matplotlib.pyplot as plt

        target = self._normalize_plot_path(path)
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
        return self._save_plot_figure(fig, target)

    def save_summoned_units_per_episode_plot(self, path: str | os.PathLike[str]) -> Path | None:
        if not self.episode_summoned_units:
            return None

        import matplotlib.pyplot as plt

        target = self._normalize_plot_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        episodes = np.arange(1, len(self.episode_summoned_units) + 1)
        summoned_counts = np.asarray(self.episode_summoned_units, dtype=float)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(
            episodes,
            summoned_counts,
            color="#7C2D12",
            linewidth=1.8,
            label="Summoned units per episode",
        )
        ax.fill_between(episodes, summoned_counts, color="#FDBA74", alpha=0.28)

        if len(summoned_counts) >= 5:
            window = min(25, len(summoned_counts))
            kernel = np.ones(window, dtype=float) / float(window)
            rolling = np.convolve(summoned_counts, kernel, mode="valid")
            rolling_x = episodes[window - 1 :]
            ax.plot(
                rolling_x,
                rolling,
                color="#1F2937",
                linewidth=2.0,
                label=f"Rolling mean ({window})",
            )

        ax.set_title("Summoned Units Per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Units summoned")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="upper right")
        fig.tight_layout()
        return self._save_plot_figure(fig, target)

    def save_cumulative_summoned_units_plot(self, path: str | os.PathLike[str]) -> Path | None:
        if not self.episode_summoned_units:
            return None

        import matplotlib.pyplot as plt

        target = self._normalize_plot_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        episodes = np.arange(1, len(self.episode_summoned_units) + 1)
        cumulative_summoned = np.cumsum(np.asarray(self.episode_summoned_units, dtype=float))

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(
            episodes,
            cumulative_summoned,
            color="#0F766E",
            linewidth=2.1,
            label="Cumulative summoned units",
        )
        ax.fill_between(episodes, cumulative_summoned, color="#99F6E4", alpha=0.24)
        ax.set_title("Cumulative Summoned Units Over Training")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total units summoned")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="upper left")
        fig.tight_layout()
        return self._save_plot_figure(fig, target)

    def save_hired_units_per_episode_plot(self, path: str | os.PathLike[str]) -> Path | None:
        if not self.episode_hired_units:
            return None

        import matplotlib.pyplot as plt

        target = self._normalize_plot_path(path)
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
        return self._save_plot_figure(fig, target)

    def save_unit_swaps_per_episode_plot(self, path: str | os.PathLike[str]) -> Path | None:
        if not self.episode_unit_swaps:
            return None

        import matplotlib.pyplot as plt

        target = self._normalize_plot_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        episodes = np.arange(1, len(self.episode_unit_swaps) + 1)
        swap_counts = np.asarray(self.episode_unit_swaps, dtype=float)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(
            episodes,
            swap_counts,
            color="#2563EB",
            linewidth=1.8,
            label="Unit swaps per episode",
        )
        ax.fill_between(episodes, swap_counts, color="#93C5FD", alpha=0.28)

        if len(swap_counts) >= 5:
            window = min(25, len(swap_counts))
            kernel = np.ones(window, dtype=float) / float(window)
            rolling = np.convolve(swap_counts, kernel, mode="valid")
            rolling_x = episodes[window - 1 :]
            ax.plot(
                rolling_x,
                rolling,
                color="#1F2937",
                linewidth=2.0,
                label=f"Rolling mean ({window})",
            )

        ax.set_title("Unit Position Swaps Per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Swaps")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="upper right")
        fig.tight_layout()
        return self._save_plot_figure(fig, target)

    def save_battle_item_activity_plot(self, path: str | os.PathLike[str]) -> Path | None:
        if not self.episode_battle_items_equipped and not self.episode_battle_items_used:
            return None

        import matplotlib.pyplot as plt

        equip_count = len(self.episode_battle_items_equipped)
        use_count = len(self.episode_battle_items_used)
        episode_count = max(equip_count, use_count)
        if episode_count <= 0:
            return None

        target = self._normalize_plot_path(path)
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
        return self._save_plot_figure(fig, target)

    def save_scripted_bot_victories_per_episode_plot(
        self,
        path: str | os.PathLike[str],
    ) -> Path | None:
        if not self.episode_scripted_bot_victories:
            return None

        import matplotlib.pyplot as plt

        target = self._normalize_plot_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        episodes = np.arange(1, len(self.episode_scripted_bot_victories) + 1)
        victory_counts = np.asarray(self.episode_scripted_bot_victories, dtype=float)
        cumulative_victories = np.cumsum(victory_counts)

        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 8.0), sharex=True)

        ax_top.plot(
            episodes,
            victory_counts,
            color="#6D28D9",
            linewidth=1.8,
            label="Bot victories per episode",
        )
        ax_top.fill_between(episodes, victory_counts, color="#C4B5FD", alpha=0.28)
        if len(victory_counts) >= 5:
            window = min(25, len(victory_counts))
            kernel = np.ones(window, dtype=float) / float(window)
            rolling = np.convolve(victory_counts, kernel, mode="valid")
            rolling_x = episodes[window - 1 :]
            ax_top.plot(
                rolling_x,
                rolling,
                color="#312E81",
                linewidth=2.0,
                linestyle="--",
                label=f"Rolling mean ({window})",
            )
        ax_top.set_title("Scripted Capital Bot Victories")
        ax_top.set_ylabel("Enemies defeated")
        ax_top.grid(True, alpha=0.25, linewidth=0.6)
        ax_top.legend(loc="upper right")

        ax_bottom.plot(
            episodes,
            cumulative_victories,
            color="#0F766E",
            linewidth=2.1,
            label="Cumulative bot victories",
        )
        ax_bottom.fill_between(episodes, cumulative_victories, color="#99F6E4", alpha=0.22)
        ax_bottom.set_xlabel("Episode")
        ax_bottom.set_ylabel("Total enemies defeated")
        ax_bottom.grid(True, alpha=0.25, linewidth=0.6)
        ax_bottom.legend(loc="upper left")

        fig.tight_layout()
        return self._save_plot_figure(fig, target)

    def save_cumulative_hired_units_plot(self, path: str | os.PathLike[str]) -> Path | None:
        if not self.episode_hired_units:
            return None

        import matplotlib.pyplot as plt

        target = self._normalize_plot_path(path)
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
        return self._save_plot_figure(fig, target)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Campaign Agent")
    parser.add_argument("--total-steps", type=int, default=2_000_000, help="Total training steps")
    parser.add_argument("--n-envs", type=int, default=12, help="Number of parallel environments")
    parser.add_argument(
        "--vec-env",
        choices=("subproc", "dummy"),
        default="subproc",
        help="Vectorized rollout environment backend",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument(
        "--wall-time-seconds",
        type=float,
        default=None,
        help="Stop learning cleanly after this many real-time seconds and save the final model",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=500_000, help="Checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Evaluation frequency")
    parser.add_argument(
        "--stochastic-eval-episodes",
        type=int,
        default=0,
        help="Run stochastic evaluation for this many episodes; disabled by default",
    )
    parser.add_argument(
        "--no-vec-check-nan",
        action="store_true",
        help="Disable VecCheckNan wrappers for faster training",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=int(os.getenv("TORCH_NUM_THREADS", "1") or "1"),
        help="Torch/OpenMP thread limit for each process",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Disable numerics diagnostics callback and crash reports",
    )
    parser.add_argument(
        "--no-freeze-action-layout",
        action="store_true",
        help="Recompute dynamic action layout on every reset instead of freezing it for this run",
    )
    parser.add_argument(
        "--no-scripted-bot",
        action="store_true",
        help="Disable the scripted capital bot completely for faster training",
    )
    parser.add_argument(
        "--no-empire-territory",
        action="store_true",
        help="Disable Empire territory expansion and its observation layer for faster training",
    )
    parser.add_argument(
        "--dragon",
        action="store_true",
        help="Use the green dragon as the campaign objective instead of the two objective cities",
    )
    parser.add_argument(
        "--blue-dragon",
        action="store_true",
        help="Same as --dragon, but the objective dragon is the blue (water) dragon instead of the green one",
    )
    parser.add_argument(
        "--ppo-n-steps",
        type=int,
        default=2048,
        help="Rollout steps per environment before each PPO update",
    )
    parser.add_argument(
        "--ppo-batch-size",
        type=int,
        default=512,
        help="PPO minibatch size; larger values reduce optimizer minibatches",
    )
    parser.add_argument(
        "--ppo-n-epochs",
        type=int,
        default=5,
        help="PPO optimization epochs per rollout; lower values reduce update cost",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO policy/value clipping range",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="PPO entropy coefficient",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.995,
        help="Discount factor used by PPO and VecNormalize reward returns",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Optional PPO target KL for early stopping optimizer epochs",
    )
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
    parser.add_argument(
        "--behavior-report-path",
        type=str,
        default=None,
        help="Optional JSONL file for periodic campaign behavior snapshots",
    )
    parser.add_argument(
        "--behavior-story-path",
        type=str,
        default=None,
        help="Optional Markdown file for periodic free-form campaign behavior reports",
    )
    parser.add_argument(
        "--resume-model",
        type=str,
        default=None,
        help="Optional MaskablePPO .zip model path to continue training from",
    )
    parser.add_argument(
        "--resume-vecnormalize",
        type=str,
        default=None,
        help="Optional VecNormalize .pkl path to restore normalization statistics",
    )
    parser.add_argument(
        "--resume-total-steps-is-target",
        action="store_true",
        help="When resuming, interpret --total-steps as the desired final absolute timestep",
    )
    args = parser.parse_args()

    if args.ppo_n_steps < 1:
        parser.error("--ppo-n-steps must be >= 1")
    if args.ppo_batch_size < 2:
        parser.error("--ppo-batch-size must be >= 2")
    if args.ppo_n_epochs < 1:
        parser.error("--ppo-n-epochs must be >= 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be > 0")
    if args.clip_range <= 0:
        parser.error("--clip-range must be > 0")
    if args.ent_coef < 0:
        parser.error("--ent-coef must be >= 0")
    if not 0 < args.gamma <= 1:
        parser.error("--gamma must be in (0, 1]")
    if not 0 < args.gae_lambda <= 1:
        parser.error("--gae-lambda must be in (0, 1]")
    if args.target_kl is not None and args.target_kl <= 0:
        parser.error("--target-kl must be > 0 when provided")
    if args.wall_time_seconds is not None and args.wall_time_seconds <= 0:
        parser.error("--wall-time-seconds must be > 0 when provided")
    if args.resume_model is not None and not Path(args.resume_model).exists():
        parser.error(f"--resume-model does not exist: {args.resume_model}")
    if args.resume_vecnormalize is not None and not Path(args.resume_vecnormalize).exists():
        parser.error(f"--resume-vecnormalize does not exist: {args.resume_vecnormalize}")

    TORCH_NUM_THREADS = max(1, int(args.torch_num_threads))
    for thread_env_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ[thread_env_name] = str(TORCH_NUM_THREADS)
    th.set_num_threads(TORCH_NUM_THREADS)
    try:
        th.set_num_interop_threads(max(1, min(TORCH_NUM_THREADS, 2)))
    except RuntimeError:
        pass

    print("Проверка среды CampaignEnv...")
    FREEZE_DYNAMIC_ACTION_LAYOUT = not bool(args.no_freeze_action_layout)
    SCRIPTED_CAPITAL_BOT_ENABLED = not bool(args.no_scripted_bot)
    EMPIRE_TERRITORY_ENABLED = not bool(args.no_empire_territory)
    if bool(args.blue_dragon):
        CAMPAIGN_OBJECTIVE = "blue_dragon"
    elif bool(args.dragon):
        CAMPAIGN_OBJECTIVE = "dragon"
    else:
        CAMPAIGN_OBJECTIVE = "cities"

    test_env = CampaignEnv(
        grid_size=DEFAULT_GRID_SIZE,
        log_enabled=False,
        detailed_step_info=False,
        freeze_dynamic_action_layout=FREEZE_DYNAMIC_ACTION_LAYOUT,
        scripted_capital_bot_enabled=SCRIPTED_CAPITAL_BOT_ENABLED,
        empire_territory_enabled=EMPIRE_TERRITORY_ENABLED,
        campaign_objective=CAMPAIGN_OBJECTIVE,
        Realcapital=2,
        **REWARD_CONFIG,
    )
    check_env(test_env, warn=True)
    print("Среда прошла проверку!")

    TOTAL_STEPS = args.total_steps
    N_ENVS = args.n_envs
    VEC_ENV_MODE = args.vec_env
    SEED = args.seed
    CHECKPOINT_FREQ = args.checkpoint_freq
    EVAL_FREQ = args.eval_freq
    EVAL_EPISODES = 20
    STOCHASTIC_EVAL_EPISODES = max(0, int(args.stochastic_eval_episodes))
    USE_VEC_CHECK_NAN = not bool(args.no_vec_check_nan)
    USE_DIAGNOSTICS = not bool(args.no_diagnostics)
    PPO_N_STEPS = int(args.ppo_n_steps)
    PPO_BATCH_SIZE = int(args.ppo_batch_size)
    PPO_N_EPOCHS = int(args.ppo_n_epochs)
    PPO_LEARNING_RATE = float(args.learning_rate)
    PPO_CLIP_RANGE = float(args.clip_range)
    PPO_ENT_COEF = float(args.ent_coef)
    PPO_GAMMA = float(args.gamma)
    PPO_GAE_LAMBDA = float(args.gae_lambda)
    PPO_TARGET_KL = None if args.target_kl is None else float(args.target_kl)
    PPO_ROLLOUT_SIZE = PPO_N_STEPS * N_ENVS
    PPO_MINIBATCHES_PER_EPOCH = (PPO_ROLLOUT_SIZE + PPO_BATCH_SIZE - 1) // PPO_BATCH_SIZE
    PPO_GRADIENT_BATCHES_PER_ROLLOUT = PPO_N_EPOCHS * PPO_MINIBATCHES_PER_EPOCH
    VECNORM_NORM_OBS = False
    VECNORM_NORM_REWARD = True
    VECNORM_CLIP_REWARD = 10.0

    COMET_AVAILABLE = Experiment is not None and OfflineExperiment is not None and not args.no_comet
    if COMET_AVAILABLE and not _comet_version_is_supported():
        detected = getattr(comet_ml, "__version__", "unknown")
        required = ".".join(str(v) for v in REQUIRED_COMET_VERSION)
        print(f"[WARN] comet_ml>={required} is required, found {detected}. Comet logging disabled.")
        COMET_AVAILABLE = False

    run_name_suffix = {"dragon": "-dragon", "blue_dragon": "-bluedragon"}.get(
        CAMPAIGN_OBJECTIVE, ""
    )
    run_name = args.run_name or f"campaign-ppo-{TOTAL_STEPS // 1000}k{run_name_suffix}"

    comet_experiment = None
    comet_config = {
            "algo": "MaskablePPO",
            "total_timesteps": TOTAL_STEPS,
            "n_envs": N_ENVS,
            "vec_env": VEC_ENV_MODE,
            "vec_env_start_method": "spawn" if VEC_ENV_MODE == "subproc" else None,
            "seed": SEED,
            "n_steps": PPO_N_STEPS,
            "batch_size": PPO_BATCH_SIZE,
            "gamma": PPO_GAMMA,
            "gae_lambda": PPO_GAE_LAMBDA,
            "n_epochs": PPO_N_EPOCHS,
            "ppo_rollout_size": PPO_ROLLOUT_SIZE,
            "ppo_minibatches_per_epoch": PPO_MINIBATCHES_PER_EPOCH,
            "ppo_gradient_batches_per_rollout": PPO_GRADIENT_BATCHES_PER_ROLLOUT,
            "learning_rate": PPO_LEARNING_RATE,
            "clip_range": PPO_CLIP_RANGE,
            "ent_coef": PPO_ENT_COEF,
            "target_kl": PPO_TARGET_KL,
            "checkpoint_freq": CHECKPOINT_FREQ,
            "eval_freq": EVAL_FREQ,
            "comet_log_freq": args.comet_log_freq,
            "eval_episodes": EVAL_EPISODES,
            "stochastic_eval_episodes": STOCHASTIC_EVAL_EPISODES,
            "max_eval_episode_steps": MAX_EVAL_EPISODE_STEPS,
            "campaign_objective": CAMPAIGN_OBJECTIVE,
            "dragon_objective": bool(args.dragon),
            "behavior_report_path": args.behavior_report_path,
            "behavior_story_path": args.behavior_story_path,
            "comet_window_episodes": args.comet_window_episodes,
            "comet_offline": bool(args.comet_offline),
            "vec_check_nan": bool(USE_VEC_CHECK_NAN),
            "diagnostics": bool(USE_DIAGNOSTICS),
            "freeze_dynamic_action_layout": bool(FREEZE_DYNAMIC_ACTION_LAYOUT),
            "scripted_capital_bot_enabled": bool(SCRIPTED_CAPITAL_BOT_ENABLED),
            "empire_territory_enabled": bool(EMPIRE_TERRITORY_ENABLED),
            "torch_num_threads": TORCH_NUM_THREADS,
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

    CHECKPOINT_DIR = str(BASE_DIR / "campaign_checkpoints" / run_id)
    MODEL_DIR = f"./campaign_models/{run_id}"
    EVAL_DIR = f"./campaign_eval/{run_id}"
    TB_LOG_DIR = f"./campaign_tb_logs/{run_id}"
    DIAGNOSTICS_DIR = f"./campaign_diagnostics/{run_id}" if USE_DIAGNOSTICS else None
    COMET_OFFLINE_DIR = args.comet_offline_dir

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)
    if DIAGNOSTICS_DIR is not None:
        os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)
    os.makedirs(COMET_OFFLINE_DIR, exist_ok=True)

    if COMET_AVAILABLE and comet_experiment is not None:
        comet_experiment.log_parameters(
            {
                "run_id": run_id,
                "checkpoint_dir": CHECKPOINT_DIR,
                "model_dir": MODEL_DIR,
                "eval_dir": EVAL_DIR,
                "tb_log_dir": TB_LOG_DIR,
                "diagnostics_dir": DIAGNOSTICS_DIR or "disabled",
                "comet_offline_dir": COMET_OFFLINE_DIR,
            },
            prefix="paths",
        )

    vec_env_cls, vec_env_kwargs = resolve_training_vec_env_config(VEC_ENV_MODE)
    env_factory = partial(
        make_env,
        freeze_dynamic_action_layout=FREEZE_DYNAMIC_ACTION_LAYOUT,
        scripted_capital_bot_enabled=SCRIPTED_CAPITAL_BOT_ENABLED,
        empire_territory_enabled=EMPIRE_TERRITORY_ENABLED,
        campaign_objective=CAMPAIGN_OBJECTIVE,
    )
    print(f"Создание {N_ENVS} параллельных сред ({VEC_ENV_MODE})...")
    vec_env = make_vec_env(
        env_factory,
        n_envs=N_ENVS,
        seed=SEED,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )
    if args.resume_vecnormalize is not None:
        vec_env = VecNormalize.load(args.resume_vecnormalize, vec_env)
        vec_env.training = True
        vec_env.norm_reward = VECNORM_NORM_REWARD
        print(f"[Resume] VecNormalize stats loaded: {args.resume_vecnormalize}")
    else:
        vec_env = VecNormalize(
            vec_env,
            training=True,
            norm_obs=VECNORM_NORM_OBS,
            norm_reward=VECNORM_NORM_REWARD,
            clip_reward=VECNORM_CLIP_REWARD,
            gamma=PPO_GAMMA,
        )
    if USE_VEC_CHECK_NAN:
        vec_env = VecCheckNan(
            vec_env,
            raise_exception=True,
            warn_once=False,
            check_inf=True,
        )

    eval_env = DummyVecEnv(
        [
            lambda: make_env(
                log_enabled=False,
                eval_max_episode_steps=MAX_EVAL_EPISODE_STEPS,
                freeze_dynamic_action_layout=FREEZE_DYNAMIC_ACTION_LAYOUT,
                scripted_capital_bot_enabled=SCRIPTED_CAPITAL_BOT_ENABLED,
                empire_territory_enabled=EMPIRE_TERRITORY_ENABLED,
                campaign_objective=CAMPAIGN_OBJECTIVE,
            )
        ]
    )
    if args.resume_vecnormalize is not None:
        eval_env = VecNormalize.load(args.resume_vecnormalize, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(
            eval_env,
            training=False,
            norm_obs=VECNORM_NORM_OBS,
            norm_reward=False,
            clip_reward=VECNORM_CLIP_REWARD,
            gamma=PPO_GAMMA,
        )
    if USE_VEC_CHECK_NAN:
        eval_env = VecCheckNan(
            eval_env,
            raise_exception=True,
            warn_once=False,
            check_inf=True,
        )

    print("Инициализация MaskablePPO...")
    if args.resume_model is not None:
        model = MaskablePPO.load(
            args.resume_model,
            env=vec_env,
            device="cpu",
            verbose=1,
            tensorboard_log=TB_LOG_DIR,
        )
        print(f"[Resume] Model loaded: {args.resume_model}")
        print(f"[Resume] Model timesteps: {model.num_timesteps:,}")
    else:
        model = MaskablePPO(
            policy="MlpPolicy",
            env=vec_env,
            device="cpu",
            verbose=1,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            gae_lambda=PPO_GAE_LAMBDA,
            gamma=PPO_GAMMA,
            n_epochs=PPO_N_EPOCHS,
            learning_rate=PPO_LEARNING_RATE,
            clip_range=PPO_CLIP_RANGE,
            ent_coef=PPO_ENT_COEF,
            vf_coef=0.5,
            target_kl=PPO_TARGET_KL,
            seed=SEED,
            tensorboard_log=TB_LOG_DIR,
        )

    resume_start_timesteps = int(model.num_timesteps)
    learn_total_timesteps = TOTAL_STEPS
    reset_num_timesteps = True
    if args.resume_model is not None:
        reset_num_timesteps = False
        if args.resume_total_steps_is_target:
            learn_total_timesteps = max(0, TOTAL_STEPS - resume_start_timesteps)

    callbacks = []

    checkpoint_cb = CampaignCheckpointCallback(
        save_dir=CHECKPOINT_DIR,
        milestone_steps=CHECKPOINT_FREQ,
        start_timesteps=resume_start_timesteps,
        experiment=comet_experiment,
        verbose=1,
    )
    callbacks.append(checkpoint_cb)

    metrics_cb = CampaignMetricsCallback(
        log_freq=args.comet_log_freq,
        episode_window=args.comet_window_episodes,
        experiment=comet_experiment,
        report_path=args.behavior_report_path,
        story_path=args.behavior_story_path,
        verbose=0,
    )
    callbacks.append(metrics_cb)
    wall_clock_cb = None
    if args.wall_time_seconds is not None:
        wall_clock_cb = WallClockStopCallback(
            wall_time_seconds=args.wall_time_seconds,
            verbose=1,
        )
        callbacks.append(wall_clock_cb)
    if USE_DIAGNOSTICS and DIAGNOSTICS_DIR is not None:
        diagnostics_cb = CampaignNumericsDiagnosticsCallback(
            report_dir=DIAGNOSTICS_DIR,
            experiment=comet_experiment,
            verbose=1,
        )
        callbacks.append(diagnostics_cb)

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
        stochastic_eval_episodes=STOCHASTIC_EVAL_EPISODES,
    )
    callbacks.append(eval_cb)

    print(f"\n{'='*60}")
    print("Запуск обучения Campaign Agent")
    print(f"Всего шагов: {TOTAL_STEPS:,}")
    print(f"Параллельных сред: {N_ENVS}")
    print(f"VecEnv: {VEC_ENV_MODE}")
    if SEED is not None:
        print(f"Seed: {SEED}")
    print(f"Checkpoint каждые: {CHECKPOINT_FREQ:,} шагов")
    print(f"Evaluation каждые: {EVAL_FREQ:,} шагов")
    print(f"Eval deterministic эпизодов: {EVAL_EPISODES}")
    print(f"Eval stochastic эпизодов: {STOCHASTIC_EVAL_EPISODES}")
    print(f"Eval лимит шагов на эпизод: {MAX_EVAL_EPISODE_STEPS}")
    print(f"Campaign objective: {CAMPAIGN_OBJECTIVE}")
    print(f"PPO n_steps: {PPO_N_STEPS:,}")
    print(f"PPO batch_size: {PPO_BATCH_SIZE:,}")
    print(f"PPO n_epochs: {PPO_N_EPOCHS:,}")
    print(f"PPO learning_rate: {PPO_LEARNING_RATE:g}")
    print(f"PPO clip_range: {PPO_CLIP_RANGE:g}")
    print(f"PPO ent_coef: {PPO_ENT_COEF:g}")
    print(f"PPO gamma: {PPO_GAMMA:g}")
    print(f"PPO gae_lambda: {PPO_GAE_LAMBDA:g}")
    print(f"PPO target_kl: {PPO_TARGET_KL if PPO_TARGET_KL is not None else 'disabled'}")
    print(f"PPO gradient batches per rollout: {PPO_GRADIENT_BATCHES_PER_ROLLOUT:,}")
    print(f"VecCheckNan: {'enabled' if USE_VEC_CHECK_NAN else 'disabled'}")
    print(f"Freeze action layout: {'enabled' if FREEZE_DYNAMIC_ACTION_LAYOUT else 'disabled'}")
    print(f"Empire territory: {'enabled' if EMPIRE_TERRITORY_ENABLED else 'disabled'}")
    print(f"Torch threads per process: {TORCH_NUM_THREADS}")
    print(f"Diagnostics: {'enabled' if USE_DIAGNOSTICS else 'disabled'}")
    print(
        "Wall-clock training limit: "
        f"{args.wall_time_seconds:g}s" if args.wall_time_seconds is not None else
        "Wall-clock training limit: disabled"
    )
    if DIAGNOSTICS_DIR is not None:
        print(f"Diagnostics dir: {DIAGNOSTICS_DIR}")
    if args.resume_model is not None:
        print(f"Resume model: {args.resume_model}")
        print(f"Resume VecNormalize: {args.resume_vecnormalize or 'not provided'}")
        print(f"Resume start timesteps: {resume_start_timesteps:,}")
        print(f"Learn call timesteps: {learn_total_timesteps:,}")
        print(f"Reset num timesteps: {reset_num_timesteps}")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}\n")

    if learn_total_timesteps > 0:
        try:
            model.learn(
                total_timesteps=learn_total_timesteps,
                callback=CallbackList(callbacks),
                reset_num_timesteps=reset_num_timesteps,
            )
        except Exception as exc:
            if USE_DIAGNOSTICS and DIAGNOSTICS_DIR is not None:
                crash_report_path = Path(DIAGNOSTICS_DIR) / f"learn_crash_step_{model.num_timesteps}.txt"
                written_path = _write_training_diagnostics_report(
                    crash_report_path,
                    model=model,
                    exception=exc,
                    issues=_collect_policy_parameter_issues(model) + _collect_rollout_buffer_issues(model),
                )
                print(f"[DIAG] Training crash diagnostics written to {written_path}")
                if COMET_AVAILABLE and comet_experiment is not None:
                    comet_experiment.log_other("diagnostics_last_crash_report", str(written_path))
            raise
    else:
        print(
            "[Resume] Skipping learn(): model timesteps are already at or above "
            f"target total steps ({TOTAL_STEPS:,})."
        )

    final_model_path = f"{MODEL_DIR}/final_model"
    model.save(final_model_path)
    vec_norm = model.get_vec_normalize_env()
    if vec_norm is not None:
        vec_norm_path = f"{MODEL_DIR}/vecnormalize.pkl"
        vec_norm.save(vec_norm_path)
        print(f"VecNormalize stats saved: {vec_norm_path}")
    print(f"\nФинальная модель сохранена: {final_model_path}.zip")

    print(f"\n{'='*60}")
    print("ИТОГИ ОБУЧЕНИЯ:")
    print(f"  Побед: {metrics_cb.victories}")
    print(f"  Поражений: {metrics_cb.defeats}")
    print(f"  Таймаутов: {metrics_cb.timeouts}")
    print(f"  Побед бота столицы над врагами: {metrics_cb.total_scripted_bot_victories}")
    print(f"  Перестановок юнитов: {metrics_cb.total_unit_swaps}")
    total = metrics_cb.victories + metrics_cb.defeats + metrics_cb.timeouts
    if total > 0:
        print(f"  Winrate: {100 * metrics_cb.victories / total:.1f}%")
    print(f"{'='*60}")

    summoned_episode_plot_path = BASE_DIR / "outputs" / "campaign_summoned_units_per_episode.png"
    saved_summoned_episode_plot = metrics_cb.save_summoned_units_per_episode_plot(
        summoned_episode_plot_path
    )
    if saved_summoned_episode_plot is not None:
        print(f"  График призывов за эпизод: {saved_summoned_episode_plot}")

    summoned_total_plot_path = BASE_DIR / "outputs" / "campaign_summoned_units_cumulative.png"
    saved_summoned_total_plot = metrics_cb.save_cumulative_summoned_units_plot(
        summoned_total_plot_path
    )
    if saved_summoned_total_plot is not None:
        print(f"  График призывов за всё обучение: {saved_summoned_total_plot}")

    hired_episode_plot_path = BASE_DIR / "outputs" / "campaign_hired_units_per_episode.png"
    saved_hired_episode_plot = metrics_cb.save_hired_units_per_episode_plot(hired_episode_plot_path)
    if saved_hired_episode_plot is not None:
        print(f"  График найма за эпизод: {saved_hired_episode_plot}")

    hired_total_plot_path = BASE_DIR / "outputs" / "campaign_hired_units_cumulative.png"
    saved_hired_total_plot = metrics_cb.save_cumulative_hired_units_plot(hired_total_plot_path)
    if saved_hired_total_plot is not None:
        print(f"  График найма за всё обучение: {saved_hired_total_plot}")

    unit_swap_plot_path = BASE_DIR / "outputs" / "campaign_unit_swaps_per_episode.png"
    saved_unit_swap_plot = metrics_cb.save_unit_swaps_per_episode_plot(unit_swap_plot_path)
    if saved_unit_swap_plot is not None:
        print(f"  График перестановок юнитов за эпизод: {saved_unit_swap_plot}")

    battle_item_plot_path = BASE_DIR / "outputs" / "campaign_battle_item_activity.png"
    saved_battle_item_plot = metrics_cb.save_battle_item_activity_plot(battle_item_plot_path)
    if saved_battle_item_plot is not None:
        print(f"  График боевых предметов: {saved_battle_item_plot}")

    scripted_bot_plot_path = BASE_DIR / "outputs" / "campaign_scripted_bot_victories_per_episode.png"
    saved_scripted_bot_plot = metrics_cb.save_scripted_bot_victories_per_episode_plot(
        scripted_bot_plot_path
    )
    if saved_scripted_bot_plot is not None:
        print(f"  График побед бота столицы за эпизод: {saved_scripted_bot_plot}")

    spell_cast_plot_path = BASE_DIR / "outputs" / "campaign_spell_cast_activity.png"
    saved_spell_cast_plot = metrics_cb.save_spell_cast_activity_plot(spell_cast_plot_path)
    if saved_spell_cast_plot is not None:
        print(f"  График применения заклинаний: {saved_spell_cast_plot}")

    if COMET_AVAILABLE and comet_experiment is not None:
        try:
            comet_experiment.log_tensorboard_folder(TB_LOG_DIR)
        except Exception as exc:
            print(f"[WARN] Failed to upload TensorBoard logs to Comet: {exc}")
        if saved_summoned_episode_plot is not None and hasattr(comet_experiment, "log_image"):
            try:
                comet_experiment.log_image(
                    str(saved_summoned_episode_plot),
                    name="campaign_summoned_units_per_episode",
                )
            except Exception as exc:
                print(f"[WARN] Failed to upload summoned-per-episode plot to Comet: {exc}")
        if saved_summoned_total_plot is not None and hasattr(comet_experiment, "log_image"):
            try:
                comet_experiment.log_image(
                    str(saved_summoned_total_plot),
                    name="campaign_summoned_units_cumulative",
                )
            except Exception as exc:
                print(f"[WARN] Failed to upload cumulative-summoned plot to Comet: {exc}")
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
        if saved_unit_swap_plot is not None and hasattr(comet_experiment, "log_image"):
            try:
                comet_experiment.log_image(
                    str(saved_unit_swap_plot),
                    name="campaign_unit_swaps_per_episode",
                )
            except Exception as exc:
                print(f"[WARN] Failed to upload unit-swap plot to Comet: {exc}")
        if saved_battle_item_plot is not None and hasattr(comet_experiment, "log_image"):
            try:
                comet_experiment.log_image(
                    str(saved_battle_item_plot),
                    name="campaign_battle_item_activity",
                )
            except Exception as exc:
                print(f"[WARN] Failed to upload battle-item plot to Comet: {exc}")
        if saved_scripted_bot_plot is not None and hasattr(comet_experiment, "log_image"):
            try:
                comet_experiment.log_image(
                    str(saved_scripted_bot_plot),
                    name="campaign_scripted_bot_victories_per_episode",
                )
            except Exception as exc:
                print(f"[WARN] Failed to upload scripted-bot plot to Comet: {exc}")
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
