from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (101, 102, 103, 104, 105, 106, 107)
RESULT_COLUMNS = (
    "run",
    "seed",
    "status",
    "training_seconds",
    "process_seconds",
    "steps",
    "episodes",
    "dragon_objective_wins",
    "dragon_encounters",
    "dragon_battle_wins",
    "campaign_win_rate_percent",
    "recent_win_rate_percent",
    "battle_win_rate_percent",
    "recent_reward_mean",
    "run_id",
    "model_path",
)


@dataclass
class RunResult:
    run: int
    seed: int
    status: str
    training_seconds: float
    process_seconds: float
    steps: int = 0
    episodes: int = 0
    dragon_objective_wins: int = 0
    dragon_encounters: int = 0
    dragon_battle_wins: int = 0
    campaign_win_rate_percent: float = 0.0
    recent_win_rate_percent: float = 0.0
    battle_win_rate_percent: float = 0.0
    recent_reward_mean: float = 0.0
    run_id: str = ""
    model_path: str = ""


def python_executable() -> Path:
    venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    return venv_python if venv_python.exists() else Path(sys.executable)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    reports: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            reports.append(json.loads(raw_line))
        except json.JSONDecodeError:
            continue
    return reports


def progress_summary(run_number: int, seed: int, report: dict[str, Any]) -> str:
    metrics = dict(report.get("metrics", {}) or {})
    activity = dict(report.get("recent_activity", {}) or {})
    dragon = dict(report.get("dragon_progress", {}) or {})
    uses = dict(report.get("uses", {}) or {})
    mechanics = [
        name
        for name, key in (
            ("магия", "magic"),
            ("призывы", "summon_magic"),
            ("постройки", "buildings"),
            ("найм", "hire"),
            ("предметы/зелья", "battle_items_or_potions"),
            ("лечение", "castle_heal"),
            ("продажи", "merchant_sales"),
            ("перестановки", "unit_swaps"),
        )
        if uses.get(key)
    ]
    encounters_window = int(dragon.get("encounters_window", 0) or 0)
    wins_window = int(dragon.get("victories_window", 0) or 0)
    objective_wins = int(dragon.get("objective_victories_total", 0) or 0)
    if wins_window:
        proximity = "дракон побеждён в текущем окне"
    elif encounters_window:
        proximity = "до дракона доходит, но в текущем окне не побеждает"
    elif report.get("enemy_victories_window_top"):
        proximity = "побеждает промежуточных врагов, до дракона в окне не дошёл"
    else:
        proximity = "остаётся на ранней разведке, до дракона не дошёл"
    payload = {
        "event": "progress",
        "run": run_number,
        "seed": seed,
        "step": int(report.get("step", 0) or 0),
        "episodes": int(report.get("episodes_total", 0) or 0),
        "recent_campaign_win_rate_percent": round(
            100.0 * float(metrics.get("campaign/window/victory_rate", 0.0) or 0.0), 2
        ),
        "recent_battle_win_rate_percent": round(
            100.0 * float(metrics.get("campaign/window/battle_win_rate", 0.0) or 0.0), 2
        ),
        "recent_reward_mean": round(float(report.get("reward_mean_window", 0.0) or 0.0), 3),
        "dragon_encounters_total": int(dragon.get("encounters_total", 0) or 0),
        "dragon_wins_total": int(dragon.get("victories_total", 0) or 0),
        "dragon_objective_wins_total": objective_wins,
        "proximity": proximity,
        "active_mechanics": mechanics,
        "spell_casts_per_episode": round(float(activity.get("magic_spell_casts_mean", 0.0) or 0.0), 2),
        "unit_swaps_per_episode": round(float(activity.get("unit_swaps_mean", 0.0) or 0.0), 2),
    }
    return "PROGRESS " + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def extract_run_id(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Run ID:\s*([^\s]+)", text)
    return matches[-1] if matches else ""


def extract_training_seconds(log_path: Path) -> float:
    if not log_path.exists():
        return 0.0
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"\[WallClock\].*?after\s+([0-9.]+)s", text)
    return float(matches[-1]) if matches else 0.0


def make_result(
    run_number: int,
    seed: int,
    return_code: int,
    process_seconds: float,
    report_path: Path,
    log_path: Path,
) -> RunResult:
    reports = load_jsonl(report_path)
    report = reports[-1] if reports else {}
    metrics = dict(report.get("metrics", {}) or {})
    dragon = dict(report.get("dragon_progress", {}) or {})
    run_id = extract_run_id(log_path)
    model_path = BASE_DIR / "campaign_models" / run_id / "final_model.zip" if run_id else None
    status = "ok" if return_code == 0 and model_path is not None and model_path.exists() else f"failed({return_code})"
    return RunResult(
        run=run_number,
        seed=seed,
        status=status,
        training_seconds=round(extract_training_seconds(log_path), 3),
        process_seconds=round(process_seconds, 3),
        steps=int(report.get("step", 0) or 0),
        episodes=int(report.get("episodes_total", 0) or 0),
        dragon_objective_wins=int(dragon.get("objective_victories_total", 0) or 0),
        dragon_encounters=int(dragon.get("encounters_total", 0) or 0),
        dragon_battle_wins=int(dragon.get("victories_total", 0) or 0),
        campaign_win_rate_percent=round(
            100.0 * float(metrics.get("campaign/victory_rate", 0.0) or 0.0), 3
        ),
        recent_win_rate_percent=round(
            100.0 * float(metrics.get("campaign/window/victory_rate", 0.0) or 0.0), 3
        ),
        battle_win_rate_percent=round(
            100.0 * float(metrics.get("campaign/battle_win_rate", 0.0) or 0.0), 3
        ),
        recent_reward_mean=round(float(report.get("reward_mean_window", 0.0) or 0.0), 3),
        run_id=run_id,
        model_path=str(model_path) if model_path is not None else "",
    )


def write_tables(output_dir: Path, results: list[RunResult], objective: str) -> None:
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    lines = [
        "# Семь часовых обучений против зелёного дракона",
        "",
        "| Запуск | Seed | Статус | Обучение, с | Процесс, с | Шаги | Эпизоды | Победы над драконом | Встречи | Победы в бою | Общий winrate | Последний winrate | Battle winrate | Reward | Run ID |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    lines[0] = f"# {len(results)} completed hourly trainings against {objective} dragon"
    for result in results:
        lines.append(
            f"| {result.run} | {result.seed} | {result.status} | {result.training_seconds:.1f} | "
            f"{result.process_seconds:.1f} | {result.steps} | {result.episodes} | {result.dragon_objective_wins} | "
            f"{result.dragon_encounters} | {result.dragon_battle_wins} | "
            f"{result.campaign_win_rate_percent:.3f}% | {result.recent_win_rate_percent:.3f}% | "
            f"{result.battle_win_rate_percent:.3f}% | {result.recent_reward_mean:.3f} | "
            f"{result.run_id} |"
        )
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(
    output_dir: Path,
    run_number: int,
    seed: int,
    duration_seconds: float,
    progress_steps: int,
    objective: str,
) -> RunResult:
    run_dir = output_dir / f"run_{run_number:02d}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "behavior_report.jsonl"
    story_path = run_dir / "behavior_story.md"
    stdout_path = run_dir / "train.stdout.log"
    stderr_path = run_dir / "train.stderr.log"
    objective_flag = "--blue-dragon" if objective == "blue" else "--dragon"
    command = [
        str(python_executable()),
        "-u",
        "train_campaign.py",
        objective_flag,
        "--no-scripted-bot",
        "--total-steps",
        "1000000000",
        "--wall-time-seconds",
        str(duration_seconds),
        "--n-envs",
        "12",
        "--vec-env",
        "subproc",
        "--seed",
        str(seed),
        "--checkpoint-freq",
        str(progress_steps),
        "--eval-freq",
        "1000000000",
        "--stochastic-eval-episodes",
        "0",
        "--torch-num-threads",
        "1",
        "--comet-log-freq",
        str(progress_steps),
        "--no-comet",
        "--behavior-report-path",
        str(report_path),
        "--behavior-story-path",
        str(story_path),
        "--run-name",
        f"{objective}-dragon-hourly-{run_number:02d}-seed-{seed}",
        "--ppo-n-steps",
        "2048",
        "--ppo-batch-size",
        "512",
        "--ppo-n-epochs",
        "5",
        "--learning-rate",
        "0.0003",
        "--clip-range",
        "0.2",
        "--ent-coef",
        "0.01",
        "--gamma",
        "0.995",
        "--gae-lambda",
        "0.95",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    started_at = time.monotonic()
    print(
        "RUN_START "
        + json.dumps(
            {"run": run_number, "seed": seed, "duration_seconds": duration_seconds},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr_handle:
        stdout_handle.write("$ " + subprocess.list2cmdline(command) + "\n")
        stdout_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=BASE_DIR,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=env,
        )
        emitted_reports = 0
        while process.poll() is None:
            reports = load_jsonl(report_path)
            for report in reports[emitted_reports:]:
                print(progress_summary(run_number, seed, report), flush=True)
            emitted_reports = len(reports)
            time.sleep(10)
        return_code = process.wait()
    reports = load_jsonl(report_path)
    for report in reports[emitted_reports:]:
        print(progress_summary(run_number, seed, report), flush=True)
    elapsed_seconds = time.monotonic() - started_at
    result = make_result(
        run_number,
        seed,
        return_code,
        elapsed_seconds,
        report_path,
        stdout_path,
    )
    print("RUN_END " + json.dumps(asdict(result), ensure_ascii=False, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one-hour dragon-objective training jobs.")
    parser.add_argument("--duration-seconds", type=float, default=3600.0)
    parser.add_argument("--progress-steps", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--objective", choices=("green", "blue"), default="green")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be > 0")
    if args.progress_steps <= 0:
        parser.error("--progress-steps must be > 0")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("all seeds must be distinct")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else BASE_DIR / "outputs" / f"{args.objective}_dragon_hourly_{len(args.seeds)}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "objective": f"{args.objective}_dragon_defeated",
        "duration_seconds_per_run": args.duration_seconds,
        "progress_steps": args.progress_steps,
        "seeds": args.seeds,
        "hyperparameters": {
            "n_envs": 12,
            "vec_env": "subproc",
            "ppo_n_steps": 2048,
            "ppo_batch_size": 512,
            "ppo_n_epochs": 5,
            "learning_rate": 3e-4,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "scripted_bot": False,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    results: list[RunResult] = []
    write_tables(output_dir, results, args.objective)
    print(f"OUTPUT_DIR {output_dir}", flush=True)
    for run_number, seed in enumerate(args.seeds, start=1):
        result = run_one(
            output_dir,
            run_number,
            seed,
            args.duration_seconds,
            args.progress_steps,
            args.objective,
        )
        results.append(result)
        write_tables(output_dir, results, args.objective)
    return 0 if all(result.status == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
