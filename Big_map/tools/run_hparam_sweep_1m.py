from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "hparam_sweep_1m"
DEFAULT_TOTAL_STEPS = 1_000_000
DEFAULT_EVAL_EPISODES = 20


@dataclass(frozen=True)
class SweepRun:
    index: int
    name: str
    reason: str
    params: dict[str, Any]


BASE_PARAMS: dict[str, Any] = {
    "total_steps": DEFAULT_TOTAL_STEPS,
    "n_envs": 12,
    "vec_env": "subproc",
    "seed": 1,
    "checkpoint_freq": DEFAULT_TOTAL_STEPS,
    "eval_freq": DEFAULT_TOTAL_STEPS,
    "stochastic_eval_episodes": 0,
    "torch_num_threads": 1,
    "ppo_n_steps": 2048,
    "ppo_batch_size": 512,
    "ppo_n_epochs": 5,
    "learning_rate": 3e-4,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "target_kl": None,
    "comet_log_freq": 10000,
    "no_comet": True,
}

TRACKED_KEYS = (
    "n_envs",
    "seed",
    "ppo_n_steps",
    "ppo_batch_size",
    "ppo_n_epochs",
    "learning_rate",
    "clip_range",
    "ent_coef",
    "gamma",
    "gae_lambda",
    "target_kl",
)

PARAM_LABELS = {
    "n_envs": "число параллельных окружений",
    "seed": "seed",
    "ppo_n_steps": "длина rollout на окружение",
    "ppo_batch_size": "размер minibatch PPO",
    "ppo_n_epochs": "число PPO-эпох на rollout",
    "learning_rate": "learning rate",
    "clip_range": "PPO clip range",
    "ent_coef": "entropy bonus",
    "gamma": "discount factor gamma",
    "gae_lambda": "GAE lambda",
    "target_kl": "target KL",
}

TRAIN_METRIC_LABELS = {
    "final_training_winrate_percent": "winrate во время обучения, %",
}

EVAL_METRIC_LABELS = {
    "wins": "победы",
    "defeats": "поражения",
    "timeouts": "таймауты",
    "other": "прочие исходы",
    "mean_reward": "средний reward",
    "mean_enemies_defeated": "среднее число побежденных врагов",
}

TB_SCALAR_LABELS = {
    "rollout/ep_len_mean": "средняя длина эпизода в rollout",
    "rollout/ep_rew_mean": "средний reward в rollout",
    "time/fps": "скорость обучения, samples/sec",
    "train/approx_kl": "примерный KL между старой и новой политикой",
    "train/clip_fraction": "доля PPO-обновлений, попавших под clip",
    "train/entropy_loss": "entropy loss",
    "train/explained_variance": "качество value-функции",
    "train/loss": "общий loss PPO",
    "train/policy_gradient_loss": "policy gradient loss",
    "train/value_loss": "value loss",
    "eval/victory_rate": "victory rate внутренней eval",
    "eval/deterministic_victory_rate": "deterministic victory rate внутренней eval",
    "eval/stochastic_victory_rate": "stochastic victory rate внутренней eval",
    "campaign/window/victory_rate": "скользящий campaign victory rate",
    "campaign/window/unit_swaps_mean": "скользящее среднее unit swaps",
    "campaign/unit_swaps_per_episode_mean": "среднее swaps на эпизод",
}


def build_sweep() -> list[SweepRun]:
    runs: list[SweepRun] = []

    def add(name: str, reason: str, **overrides: Any) -> None:
        params = dict(BASE_PARAMS)
        params.update(overrides)
        runs.append(
            SweepRun(
                index=len(runs) + 1,
                name=name,
                reason=reason,
                params=params,
            )
        )

    add("baseline_seed1", "Опорный конфиг после увеличения штрафа за unit swap в 2 раза.", seed=1)
    add("baseline_seed2", "Повтор опорного конфига с другим seed, чтобы измерить шум от случайности.", seed=2)
    add("baseline_seed3", "Еще один повтор опорного конфига: если seed сильно меняет результат, конфиг нестабилен.", seed=3)

    add("lr2e4_ent010", "Снижаем learning rate при прежнем давлении exploration, чтобы проверить, помогает ли более мягкий update.", learning_rate=2e-4)
    add("lr1e4_ent010", "Сильно консервативный learning rate: проверка гипотезы, что политика разваливается от слишком резких PPO-обновлений.", learning_rate=1e-4)
    add("lr3e4_ent003", "Оставляем learning rate, но снижаем entropy bonus, чтобы уменьшить бесцельное исследование и спам действий.", ent_coef=0.003)
    add("lr2e4_ent003", "Баланс стабильности: одновременно ниже learning rate и ниже entropy bonus.", learning_rate=2e-4, ent_coef=0.003)
    add("lr1e4_ent003", "Очень стабильный update с умеренным exploration: кандидат, если baseline шумит или деградирует.", learning_rate=1e-4, ent_coef=0.003)
    add("lr3e4_ent001", "Почти убираем entropy pressure при текущем learning rate, чтобы проверить, мешает ли exploration доводить кампанию до победы.", ent_coef=0.001)
    add("lr2e4_ent001", "Ниже learning rate и почти нет entropy pressure: кандидат против action spam.", learning_rate=2e-4, ent_coef=0.001)
    add("lr1e4_ent001", "Самая консервативная пара learning rate / entropy среди основных кандидатов.", learning_rate=1e-4, ent_coef=0.001)
    add("lr2e4_ent000", "Полностью отключаем явный entropy bonus, чтобы проверить, не он ли провоцирует лишние swap/движения.", learning_rate=2e-4, ent_coef=0.0)
    add("lr1e4_ent000", "Максимально консервативный no-entropy вариант: меньше риска развала, но выше риск недоисследования.", learning_rate=1e-4, ent_coef=0.0)
    add("clip015_lr2e4_ent003", "Уменьшаем clip range, чтобы ограничить скачки политики между итерациями PPO.", learning_rate=2e-4, ent_coef=0.003, clip_range=0.15)
    add("clip010_lr2e4_ent003", "Строгий clip range: проверка максимальной стабильности ценой более медленного обучения.", learning_rate=2e-4, ent_coef=0.003, clip_range=0.10)

    add("batch1024_epochs3", "Больше minibatch и меньше эпох: меньше переиспользования одного rollout, updates должны быть спокойнее.", ppo_batch_size=1024, ppo_n_epochs=3)
    add("batch1024_epochs5", "Больше minibatch при текущем числе эпох: проверяем, снижает ли это шум градиента без замедления обучения.", ppo_batch_size=1024)
    add("batch1024_epochs8", "Больше minibatch и больше эпох: проверяем, выдерживает ли среда более сильное дообучение на rollout.", ppo_batch_size=1024, ppo_n_epochs=8)
    add("batch2048_epochs3", "Очень крупный minibatch с малым числом эпох: максимально сглаженный update.", ppo_batch_size=2048, ppo_n_epochs=3)
    add("batch2048_epochs5", "Очень крупный minibatch при текущем числе эпох: проверка стабильности без уменьшения data reuse.", ppo_batch_size=2048)
    add("batch2048_epochs8", "Крупный minibatch и высокий data reuse: stress test на переобучение PPO-итерации.", ppo_batch_size=2048, ppo_n_epochs=8)
    add("batch512_epochs3", "Текущий batch size, но меньше эпох: изолируем влияние ppo_n_epochs.", ppo_n_epochs=3)
    add("batch512_epochs8", "Текущий batch size, но больше эпох: проверяем, помогает ли глубже выжимать каждый rollout.", ppo_n_epochs=8)

    add("env8_steps2048", "Меньше параллельных env: rollout распределение может стать менее шумным, но sample throughput ниже.", n_envs=8)
    add("env8_steps1024", "Меньше env и короче rollout: более частые PPO updates, полезно если политика быстро уходит в плохую область.", n_envs=8, ppo_n_steps=1024)
    add("env16_steps2048", "Больше env: шире покрытие состояний кампании в одном rollout.", n_envs=16, ppo_batch_size=1024)
    add("env16_steps1024", "Больше env и короче per-env rollout: сохраняем широкий сбор состояний, но чаще обновляем политику.", n_envs=16, ppo_n_steps=1024, ppo_batch_size=1024)
    add("env12_steps4096", "Длиннее rollout: больше контекста для delayed campaign rewards и меньше дисперсия advantages.", ppo_n_steps=4096, ppo_batch_size=1024)
    add("env8_steps4096", "Длинный rollout при меньшем числе env: проверка, лучше ли агент учится на более длинных траекториях.", n_envs=8, ppo_n_steps=4096, ppo_batch_size=1024)

    add("gamma099_gae095", "Ниже gamma: длинные эпизоды и swap-heavy поведение должны штрафоваться быстрее.", gamma=0.99)
    add("gamma0992_gae092", "Ниже gamma и GAE lambda: меньше variance в advantages, но короче credit assignment.", gamma=0.992, gae_lambda=0.92)
    add("gamma0997_gae095", "Выше gamma: проверка, нужны ли агенту более дальние delayed rewards для победы в кампании.", gamma=0.997)
    add("gamma0995_gae090", "Текущий gamma, но ниже GAE lambda: более стабильные advantages без изменения горизонта reward.", gae_lambda=0.90)

    if len(runs) != 33:
        raise RuntimeError(f"Expected 33 sweep runs, got {len(runs)}")
    return runs


def python_exe() -> str:
    venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def arg_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def build_train_command(run: SweepRun, output_dir: Path) -> list[str]:
    cmd = [python_exe(), "-u", "train_campaign.py"]
    params = dict(run.params)
    params["run_name"] = f"hpsweep1m-{run.index:02d}-{run.name}"
    for key, value in params.items():
        if value is None:
            continue
        flag = arg_name(key)
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        cmd.extend([flag, str(value)])
    return cmd


def build_eval_command(run_id: str, episodes: int) -> list[str]:
    return [
        python_exe(),
        "-u",
        "bestmodeltest.py",
        "--run-id",
        run_id,
        "--model-type",
        "final",
        "--episodes",
        str(int(episodes)),
        "--deterministic",
    ]


def render_command(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


def run_command(cmd: list[str], log_path: Path, *, timeout: int | None = None) -> tuple[int, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_parts: list[str] = []
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"$ {render_command(cmd)}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        start = time.monotonic()
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                output_parts.append(line)
                log_file.write(line)
                log_file.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
                if timeout is not None and time.monotonic() - start > timeout:
                    proc.kill()
                    line = f"\n[TIMEOUT] killed after {timeout} seconds\n"
                    output_parts.append(line)
                    log_file.write(line)
                    break
            return_code = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
        return return_code, "".join(output_parts)


def parse_run_id(text: str) -> str | None:
    patterns = (
        r"Run ID:\s*([^\s]+)",
        r"ID запуска:\s*([^\s]+)",
        r"campaign_models[/\\]([^/\\\s]+)[/\\]final_model",
        r"campaign_models[/\\]([^/\\\s]+)[/\\]vecnormalize",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return str(matches[-1]).strip()
    return None


def latest_run_id_after(start_time: float) -> str | None:
    candidates: list[Path] = []
    for root_name in ("campaign_models", "campaign_tb_logs"):
        root = BASE_DIR / root_name
        if root.exists():
            candidates.extend(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        return None
    candidates = [path for path in candidates if path.stat().st_mtime >= start_time - 5]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).name


def parse_eval_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    patterns: dict[str, str] = {
        "mean_reward": r"mean_reward:\s*([-+0-9.eE]+)",
        "mean_enemies_defeated": r"mean_enemies_defeated:\s*([-+0-9.eE]+)",
        "wins": r"wins:\s*(\d+)",
        "defeats": r"defeats:\s*(\d+)",
        "timeouts": r"timeouts:\s*(\d+)",
        "other": r"other:\s*(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            continue
        raw = match.group(1)
        if key in {"wins", "defeats", "timeouts", "other"}:
            metrics[key] = int(raw)
        else:
            metrics[key] = float(raw)
    return metrics


def parse_training_summary(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    winrate_matches = re.findall(r"Winrate:\s*([-+0-9.eE]+)%", text)
    if winrate_matches:
        metrics["final_training_winrate_percent"] = float(winrate_matches[-1])
    return metrics


def load_last_tb_scalars(run_id: str) -> dict[str, float]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return {}

    event_files = sorted((BASE_DIR / "campaign_tb_logs" / run_id).rglob("events.out.tfevents.*"))
    if not event_files:
        return {}

    tags = (
        "rollout/ep_len_mean",
        "rollout/ep_rew_mean",
        "time/fps",
        "train/approx_kl",
        "train/clip_fraction",
        "train/entropy_loss",
        "train/explained_variance",
        "train/loss",
        "train/policy_gradient_loss",
        "train/value_loss",
        "eval/victory_rate",
        "eval/deterministic_victory_rate",
        "eval/stochastic_victory_rate",
        "campaign/window/victory_rate",
        "campaign/window/unit_swaps_mean",
        "campaign/unit_swaps_per_episode_mean",
    )
    result: dict[str, float] = {}
    for event_file in event_files:
        try:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            available = set(accumulator.Tags().get("scalars", []))
            for tag in tags:
                if tag not in available:
                    continue
                values = accumulator.Scalars(tag)
                if values:
                    result[tag] = float(values[-1].value)
        except Exception:
            continue
    return result


def changed_params(params: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    changes: dict[str, tuple[Any, Any]] = {}
    for key in TRACKED_KEYS:
        current = params.get(key)
        baseline = BASE_PARAMS.get(key)
        if current != baseline:
            changes[key] = (baseline, current)
    return changes


def format_duration(seconds: float) -> str:
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} ч {minutes} мин {sec} с"
    if minutes:
        return f"{minutes} мин {sec} с"
    return f"{sec} с"


def fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def make_decision(eval_metrics: dict[str, Any], train_return_code: int, eval_return_code: int | None) -> str:
    if train_return_code != 0:
        return "ОТБРОСИТЬ: обучение завершилось с ошибкой"
    if eval_return_code is None:
        return "ОТБРОСИТЬ: финальная модель не была оценена"
    if eval_return_code != 0:
        return "ОТБРОСИТЬ: eval завершился с ошибкой"
    wins = int(eval_metrics.get("wins", 0) or 0)
    timeouts = int(eval_metrics.get("timeouts", 0) or 0)
    mean_enemies = float(eval_metrics.get("mean_enemies_defeated", 0.0) or 0.0)
    if wins >= 3:
        return "ОСТАВИТЬ: на eval есть несколько побед"
    if wins >= 1:
        return "ПРОВЕРИТЬ: на eval есть хотя бы одна победа"
    if mean_enemies >= 20.0 and timeouts <= max(1, DEFAULT_EVAL_EPISODES // 2):
        return "ПРОВЕРИТЬ: побед нет, но агент стабильно зачищает много врагов"
    return "ОТБРОСИТЬ: побед на eval нет, сигнал прохождения кампании слабый"


def build_conclusion(
    decision: str,
    eval_metrics: dict[str, Any],
    train_metrics: dict[str, Any],
    tb_scalars: dict[str, float],
) -> list[str]:
    lines: list[str] = []
    wins = int(eval_metrics.get("wins", 0) or 0)
    defeats = int(eval_metrics.get("defeats", 0) or 0)
    timeouts = int(eval_metrics.get("timeouts", 0) or 0)
    other = int(eval_metrics.get("other", 0) or 0)
    episodes = wins + defeats + timeouts + other
    if episodes:
        winrate = 100.0 * wins / episodes
        lines.append(
            f"На deterministic eval: {wins}/{episodes} побед ({winrate:.1f}%), "
            f"поражений {defeats}, таймаутов {timeouts}, прочих исходов {other}."
        )
    if "mean_reward" in eval_metrics or "mean_enemies_defeated" in eval_metrics:
        reward = eval_metrics.get("mean_reward", "нет данных")
        enemies = eval_metrics.get("mean_enemies_defeated", "нет данных")
        lines.append(
            "Eval-поведение: "
            f"средний reward {fmt_value(reward)}, "
            f"среднее число побежденных врагов {fmt_value(enemies)}."
        )
    if "final_training_winrate_percent" in train_metrics:
        lines.append(
            "Внутренний winrate из train-лога: "
            f"{fmt_value(train_metrics['final_training_winrate_percent'])}%. "
            "Эта цифра полезна как диагностический сигнал, но решение принимается по отдельному deterministic eval."
        )
    if "rollout/ep_rew_mean" in tb_scalars or "rollout/ep_len_mean" in tb_scalars:
        lines.append(
            "Последний rollout: "
            f"ep_rew_mean={fmt_value(tb_scalars.get('rollout/ep_rew_mean', 'нет данных'))}, "
            f"ep_len_mean={fmt_value(tb_scalars.get('rollout/ep_len_mean', 'нет данных'))}. "
            "Высокий rollout reward без побед на eval означает, что агент может получать промежуточные награды, но не завершать кампанию."
        )
    if decision.startswith("ОСТАВИТЬ"):
        lines.append("Вывод для sweep: конфиг стоит оставить в пуле кандидатов и перепроверить на дополнительных seed.")
    elif decision.startswith("ПРОВЕРИТЬ"):
        lines.append("Вывод для sweep: конфиг не лучший автоматически, но его стоит сравнить с соседними настройками и seed-повторами.")
    else:
        lines.append("Вывод для sweep: конфиг пока не выглядит перспективным; следующие runs должны искать более стабильное доведение кампании до победы.")
    return lines


def write_summary(
    *,
    run: SweepRun,
    summary_path: Path,
    train_command: list[str],
    train_return_code: int,
    train_log_path: Path,
    train_output: str,
    run_id: str | None,
    eval_command: list[str] | None,
    eval_return_code: int | None,
    eval_log_path: Path | None,
    eval_output: str,
    elapsed_seconds: float,
) -> None:
    train_metrics = parse_training_summary(train_output)
    eval_metrics = parse_eval_metrics(eval_output)
    tb_scalars = load_last_tb_scalars(run_id) if run_id else {}
    decision = make_decision(eval_metrics, train_return_code, eval_return_code)
    changes = changed_params(run.params)
    status = "завершено" if train_return_code == 0 else "ошибка обучения"

    lines: list[str] = []
    lines.append(f"# Sweep run {run.index:02d}: {run.name}")
    lines.append("")
    lines.append(f"Статус: {status}")
    lines.append(f"Решение: {decision}")
    lines.append(f"ID запуска: {run_id or 'unknown'}")
    lines.append(f"Длительность: {format_duration(elapsed_seconds)} ({elapsed_seconds:.1f} с)")
    lines.append(f"Завершено: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Короткий вывод")
    for line in build_conclusion(decision, eval_metrics, train_metrics, tb_scalars):
        lines.append(f"- {line}")
    lines.append("")
    lines.append("## Почему выбран этот конфиг")
    lines.append(run.reason)
    lines.append("")
    lines.append("## Что изменено относительно baseline")
    if changes:
        for key, (old, new) in changes.items():
            label = PARAM_LABELS.get(key, key)
            lines.append(f"- {key} ({label}): {fmt_value(old)} -> {fmt_value(new)}")
    else:
        lines.append("- Ничего: это baseline-reference для сравнения остальных runs.")
    lines.append("")
    lines.append("## Полная конфигурация")
    for key in sorted(run.params):
        label = PARAM_LABELS.get(key)
        suffix = f" ({label})" if label else ""
        lines.append(f"- {key}{suffix}: {fmt_value(run.params[key])}")
    lines.append("")
    lines.append("## Команда обучения")
    lines.append(render_command(train_command))
    lines.append(f"Код возврата обучения: {train_return_code}")
    lines.append(f"Лог обучения: {train_log_path}")
    lines.append("")
    lines.append("## Метрики обучения")
    if train_metrics:
        for key, value in train_metrics.items():
            label = TRAIN_METRIC_LABELS.get(key, key)
            lines.append(f"- {key} ({label}): {fmt_value(value)}")
    else:
        lines.append("- Финальные train-метрики не удалось распарсить из лога.")
    if tb_scalars:
        lines.append("")
        lines.append("## Последние TensorBoard scalars")
        for key in sorted(tb_scalars):
            label = TB_SCALAR_LABELS.get(key, key)
            lines.append(f"- {key} ({label}): {tb_scalars[key]:.6g}")
    else:
        lines.append("")
        lines.append("## Последние TensorBoard scalars")
        lines.append("- TensorBoard scalars не удалось прочитать.")
    lines.append("")
    lines.append("## Команда eval")
    if eval_command is None:
        lines.append("Eval пропущен: не найден run_id или финальная модель.")
    else:
        lines.append(render_command(eval_command))
        lines.append(f"Код возврата eval: {eval_return_code}")
        lines.append(f"Лог eval: {eval_log_path}")
    lines.append("")
    lines.append("## Метрики eval")
    if eval_metrics:
        for key in sorted(eval_metrics):
            label = EVAL_METRIC_LABELS.get(key, key)
            lines.append(f"- {key} ({label}): {fmt_value(eval_metrics[key])}")
    else:
        lines.append("- Eval-метрики не удалось распарсить.")
    lines.append("")
    lines.append("## Хвост raw eval-лога")
    if eval_output:
        tail = "".join(eval_output.splitlines(keepends=True)[-30:])
        lines.append("```")
        lines.append(tail.rstrip())
        lines.append("```")
    else:
        lines.append("- Eval output пустой.")
    lines.append("")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def parse_return_code_from_summary(text: str, kind: str) -> int | None:
    patterns = {
        "train": (r"Train return code:\s*(-?\d+)", r"Код возврата обучения:\s*(-?\d+)"),
        "eval": (r"Eval return code:\s*(-?\d+)", r"Код возврата eval:\s*(-?\d+)"),
    }
    for pattern in patterns[kind]:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def parse_elapsed_from_summary(text: str) -> float:
    for pattern in (r"Elapsed seconds:\s*([-+0-9.eE]+)", r"Длительность:.*?\(([-+0-9.eE]+)\s*с\)"):
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return 0.0


def rewrite_existing_summary(run: SweepRun, output_dir: Path, eval_episodes: int) -> bool:
    logs_dir = output_dir / "logs"
    summaries_dir = output_dir / "summaries"
    train_log_path = logs_dir / f"{run.index:02d}_{run.name}_train.log"
    eval_log_path = logs_dir / f"{run.index:02d}_{run.name}_eval.log"
    summary_path = summaries_dir / f"{run.index:02d}_{run.name}.txt"
    if not train_log_path.exists() or not summary_path.exists():
        print(f"[REWRITE-SKIP] {run.index:02d} {run.name}: no completed train log/summary")
        return False

    train_output = train_log_path.read_text(encoding="utf-8", errors="replace")
    eval_output = eval_log_path.read_text(encoding="utf-8", errors="replace") if eval_log_path.exists() else ""
    old_summary = summary_path.read_text(encoding="utf-8", errors="replace")
    run_id = parse_run_id(train_output) or parse_run_id(eval_output) or parse_run_id(old_summary)
    train_rc = parse_return_code_from_summary(old_summary, "train")
    if train_rc is None:
        train_rc = 0 if run_id and (BASE_DIR / "campaign_models" / run_id / "final_model.zip").exists() else 1
    eval_rc = parse_return_code_from_summary(old_summary, "eval")
    if eval_rc is None and eval_output:
        eval_rc = 0

    write_summary(
        run=run,
        summary_path=summary_path,
        train_command=build_train_command(run, output_dir),
        train_return_code=train_rc,
        train_log_path=train_log_path,
        train_output=train_output,
        run_id=run_id,
        eval_command=build_eval_command(run_id, eval_episodes) if run_id else None,
        eval_return_code=eval_rc,
        eval_log_path=eval_log_path if eval_log_path.exists() else None,
        eval_output=eval_output,
        elapsed_seconds=parse_elapsed_from_summary(old_summary),
    )
    print(f"[REWRITE] {summary_path}")
    return True


def apply_smoke_overrides(run: SweepRun, total_steps: int) -> SweepRun:
    params = dict(run.params)
    params.update(
        {
            "total_steps": total_steps,
            "n_envs": 1,
            "vec_env": "dummy",
            "checkpoint_freq": max(64, total_steps),
            "eval_freq": max(1_000_000, total_steps * 100),
            "ppo_n_steps": 64,
            "ppo_batch_size": 64,
            "ppo_n_epochs": 1,
            "no_vec_check_nan": True,
            "no_diagnostics": True,
        }
    )
    return SweepRun(index=run.index, name=f"smoke_{run.name}", reason=run.reason, params=params)


def run_one(run: SweepRun, output_dir: Path, eval_episodes: int) -> None:
    logs_dir = output_dir / "logs"
    summaries_dir = output_dir / "summaries"
    train_log_path = logs_dir / f"{run.index:02d}_{run.name}_train.log"
    eval_log_path = logs_dir / f"{run.index:02d}_{run.name}_eval.log"
    summary_path = summaries_dir / f"{run.index:02d}_{run.name}.txt"

    train_cmd = build_train_command(run, output_dir)
    start_wall = time.monotonic()
    start_epoch = time.time()
    print(f"\n=== RUN {run.index:02d}/33 {run.name} ===")
    print(render_command(train_cmd))
    train_rc, train_output = run_command(train_cmd, train_log_path)
    elapsed = time.monotonic() - start_wall
    run_id = parse_run_id(train_output) or latest_run_id_after(start_epoch)

    eval_cmd: list[str] | None = None
    eval_rc: int | None = None
    eval_output = ""
    if train_rc == 0 and run_id is not None and (BASE_DIR / "campaign_models" / run_id / "final_model.zip").exists():
        eval_cmd = build_eval_command(run_id, eval_episodes)
        print(f"\n=== EVAL {run.index:02d}/33 {run.name} run_id={run_id} ===")
        print(render_command(eval_cmd))
        eval_rc, eval_output = run_command(eval_cmd, eval_log_path)
    else:
        print(f"[WARN] Skipping eval for {run.name}: train_rc={train_rc}, run_id={run_id}")

    write_summary(
        run=run,
        summary_path=summary_path,
        train_command=train_cmd,
        train_return_code=train_rc,
        train_log_path=train_log_path,
        train_output=train_output,
        run_id=run_id,
        eval_command=eval_cmd,
        eval_return_code=eval_rc,
        eval_log_path=eval_log_path if eval_cmd is not None else None,
        eval_output=eval_output,
        elapsed_seconds=elapsed,
    )
    print(f"[SUMMARY] {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 33x1M CampaignEnv PPO hparam sweep.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument("--force", action="store_true", help="Run even when the summary file already exists.")
    parser.add_argument(
        "--rewrite-summaries-only",
        action="store_true",
        help="Rewrite already completed summary files from existing logs without running training.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running training.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Override each selected run to a tiny single-env training for validating the runner.",
    )
    args = parser.parse_args()

    if args.start_index < 1:
        parser.error("--start-index must be >= 1")
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be >= 1")
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be >= 1")
    if args.total_steps < 1:
        parser.error("--total-steps must be >= 1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "summaries").mkdir(parents=True, exist_ok=True)

    runs = build_sweep()
    selected = [run for run in runs if run.index >= args.start_index]
    if args.max_runs is not None:
        selected = selected[: args.max_runs]

    materialized: list[SweepRun] = []
    for run in selected:
        params = dict(run.params)
        params["total_steps"] = int(args.total_steps)
        params["checkpoint_freq"] = int(args.total_steps)
        params["eval_freq"] = int(args.total_steps)
        run = SweepRun(index=run.index, name=run.name, reason=run.reason, params=params)
        if args.smoke:
            run = apply_smoke_overrides(run, total_steps=min(int(args.total_steps), 128))
        materialized.append(run)

    print(f"Output dir: {output_dir}")
    print(f"Selected runs: {len(materialized)} / 33")

    if args.rewrite_summaries_only:
        rewritten = 0
        for run in materialized:
            rewritten += int(rewrite_existing_summary(run, output_dir, int(args.eval_episodes)))
        print(f"Rewritten summaries: {rewritten}")
        return 0

    for run in materialized:
        summary_path = output_dir / "summaries" / f"{run.index:02d}_{run.name}.txt"
        if summary_path.exists() and not args.force:
            print(f"[SKIP] {run.index:02d} {run.name}: summary exists at {summary_path}")
            continue
        train_cmd = build_train_command(run, output_dir)
        if args.dry_run:
            print(f"[DRY] {run.index:02d} {run.name}: {render_command(train_cmd)}")
            continue
        run_one(run, output_dir, int(args.eval_episodes))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
