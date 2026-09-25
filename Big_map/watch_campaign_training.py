"""Train a campaign and evaluate saved checkpoints at transition milestones."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from benchmark_training import ROOT, save_json


def latest_behavior(path, before=None):
    if not path.exists():
        return None
    for line in reversed(path.read_text().splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if before is None or row["step"] <= before:
            return row
    return None


def evaluation_story(evaluation, milestone, behavior=None):
    lines = [f"## Оценка на {milestone:,} шагах", "",
             f"Фактический шаг сохранённой модели: {evaluation['model_steps']:,}.", ""]
    for mode, label in (("deterministic", "Наиболее вероятное действие"),
                        ("stochastic", "Действие по вероятностям политики")):
        row = evaluation[mode]
        n = len(row["episodes"])
        activity = row["mean_activity"]
        lines += [
            (f"**{label}:** побед {row['wins']}/{n}; таймаутов {row['timeouts']}/{n}; "
             f"средняя награда {row['mean_reward']:.2f}."),
            (f"Вступал в бой максимум с **волной {row['max_reached_wave']}**; "
            f"полностью победил максимум **{row['max_waves_defeated']} волн**, "
            f"в среднем {row['mean_waves']:.2f}. "
             f"Распределение достигнутой волны по эпизодам: {row['reached_wave_counts']}."),
            ("Средние действия за эпизод: "
            f"постройки {activity['buildings']:.1f}, найм {activity['hires']:.1f}, "
            f"изучение заклинаний {activity['spells_learned']:.1f}, "
            f"применение заклинаний {activity['spell_casts']:.1f}, призванные юниты {activity['summons']:.1f}, "
            f"боевые предметы {activity['items_used']:.1f}, перестановки юнитов {activity['unit_swaps']:.1f}, "
            f"улучшения юнитов {activity['unit_upgrades']:.1f}, "
             f"сундуки {activity['chests']:.1f}, очищенные руины {activity['ruins']:.1f}."),
            (f"Посещённых клеток в среднем {row['mean_unique_positions']:.1f}; "
            f"длина эпизода {row['mean_length']:.1f} шагов. "
             f"Противники, на которых закончились эпизоды: {row['terminal_enemies']}."), "",
        ]
        if evaluation.get("objective") in ("dragon", "blue_dragon"):
            lines[-4] = (f"Встреч с драконом {row['dragon_encounters']}/{n}; "
                         f"побед над драконом {row['dragon_victories']}/{n}.")
        lines.insert(len(lines) - 1,
                     f"В среднем: возвращения в столицу {activity.get('capital_returns', 0):.2f}, "
                     f"лечение в храме {activity.get('castle_heals', 0):.2f}, "
                     f"воскрешение в храме {activity.get('castle_revives', 0):.2f}.")
    if behavior:
        activity = behavior["recent_activity"]
        lines += [(f"В обучении на срезе {behavior['step']:,} шагов "
                  f"завершено {behavior['episodes_total']} эпизодов; "
                  f"результаты: {behavior['results_total']}. "
                  f"В последних эпизодах: постройки {activity['buildings_built_mean']:.1f}, "
                  f"найм {activity['hires_mean']:.1f}, заклинания {activity['magic_spell_casts_mean']:.1f}, "
                  f"призывы {activity['summons_mean']:.1f}, перестановки {activity['unit_swaps_mean']:.1f}, "
                   f"улучшения {activity['unit_upgrades_mean']:.1f} за эпизод."), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map", dest="map_name", default="super_last_stand")
    parser.add_argument("--objective", choices=("cities", "dragon", "blue_dragon", "full_party"))
    parser.add_argument("--total-steps", type=int, default=3_000_000)
    parser.add_argument("--eval-every", type=int, default=400_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--ppo-n-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-boss-roster", action="store_true",
                        help="Use the ordinary starting party in training and evaluation")
    parser.add_argument("--after-run", help="Wait for this run to complete, then resume its final model; total steps is the cumulative target")
    args = parser.parse_args()
    if min(args.n_envs, args.torch_num_threads, args.ppo_n_steps, args.eval_episodes) <= 0:
        parser.error("environment, thread, rollout and evaluation episode counts must be positive")
    rollout_size = args.n_envs * args.ppo_n_steps
    if args.total_steps <= 0 or args.total_steps % rollout_size or args.eval_every <= 0:
        parser.error(f"total steps must be a positive multiple of {rollout_size}; eval interval must be positive")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start_steps = 0
    resume_model = resume_vec = None
    if args.after_run:
        source = Path(args.after_run).resolve()
        source_status_path = source / "launch.json"
        if not source_status_path.is_file():
            parser.error(f"source run status does not exist: {source_status_path}")
        waiting = {"status": "waiting_for_previous_run", "source_run": str(source),
                   "total_steps": args.total_steps, "n_envs": args.n_envs,
                   "torch_threads": args.torch_num_threads, "from_scratch": False,
                   "queued_at": datetime.now(timezone.utc).isoformat()}
        save_json(output / "launch.json", waiting)
        print(f"WAITING for {source} to finish training and evaluation", flush=True)
        while True:
            previous = json.loads(source_status_path.read_text())
            if previous["status"] == "completed":
                break
            if previous["status"] == "failed":
                waiting.update(status="failed", error="Source run failed")
                save_json(output / "launch.json", waiting)
                raise SystemExit("Source run failed; continuation was not started")
            time.sleep(5)
        resume_model = source / "models/final_model.zip"
        resume_vec = source / "models/vecnormalize.pkl"
        try:
            with zipfile.ZipFile(resume_model) as archive:
                model_data = json.loads(archive.read("data"))
            start_steps = int(model_data["num_timesteps"])
            if not resume_vec.is_file():
                raise ValueError("Final VecNormalize statistics are missing")
            if start_steps != previous["total_steps"]:
                raise ValueError("Final model step count does not match the completed run")
            if (int(model_data["n_envs"]), int(model_data["n_steps"])) != (args.n_envs, args.ppo_n_steps):
                raise ValueError("Resume environment count and rollout length must match the saved model")
            if args.total_steps <= start_steps or (args.total_steps - start_steps) % rollout_size:
                raise ValueError("Additional steps must be a positive multiple of the rollout size")
        except Exception as exc:
            waiting.update(status="failed", error=str(exc))
            save_json(output / "launch.json", waiting)
            raise
    log_path = output / "train.log"
    behavior_path = output / "behavior.jsonl"
    first_milestone = (start_steps // args.eval_every + 1) * args.eval_every
    milestones = list(range(first_milestone, args.total_steps + 1, args.eval_every))
    if not milestones or milestones[-1] != args.total_steps:
        milestones.append(args.total_steps)
    command = [sys.executable, "-u", str(ROOT / "train_campaign.py"),
               "--map", args.map_name, "--total-steps", str(args.total_steps),
               "--n-envs", str(args.n_envs), "--vec-env", "subproc", "--vec-start-method", "spawn",
               "--torch-num-threads", str(args.torch_num_threads), "--ppo-n-steps", str(args.ppo_n_steps),
               "--ppo-batch-size", "500", "--ppo-n-epochs", "5", "--seed", str(args.seed),
               "--no-comet", "--eval-freq", "0", "--checkpoint-freq", str(args.eval_every),
               "--run-id", output.name, "--output-root", str(output),
               "--comet-offline-dir", str(output / "comet_offline"),
               "--comet-log-freq", str(rollout_size), "--behavior-report-path", str(behavior_path)]
    objective_flags = {"dragon": "--dragon", "blue_dragon": "--blue-dragon", "full_party": "--full-party"}
    if args.objective in objective_flags:
        command.append(objective_flags[args.objective])
    if args.no_boss_roster:
        command.append("--no-boss-roster")
    if resume_model is not None:
        command += ["--resume-model", str(resume_model), "--resume-vecnormalize", str(resume_vec),
                    "--resume-total-steps-is-target"]
    status = {"status": "starting", "started_at": datetime.now(timezone.utc).isoformat(),
              "command": command, "total_steps": args.total_steps, "milestones": milestones,
              "completed_evaluations": [], "reported_training_steps": start_steps,
              "evaluation_episodes_per_mode": args.eval_episodes,
              "n_envs": args.n_envs, "torch_threads": args.torch_num_threads,
              "rollout_size": rollout_size, "from_scratch": resume_model is None,
              "start_timesteps": start_steps, "additional_steps": args.total_steps - start_steps,
              "evaluation_seed_start": 10000, "evaluation_errors": []}
    status.update(map=args.map_name, objective=args.objective)
    status["use_boss_starting_roster"] = False if args.no_boss_roster else None
    if args.after_run:
        status["source_run"] = str(source)
    save_json(output / "launch.json", status)
    (output / "progress.md").write_text(
        f"# {args.map_name} ({args.objective or 'цель карты'}): обучение на {args.total_steps:,} шагов\n\n"
        f"{args.n_envs} сред SubprocVecEnv, {args.torch_num_threads} потока PyTorch, "
        f"rollout {rollout_size:,}, batch 500, 5 эпох PPO. "
        + (f"Продолжение с {start_steps:,} шагов. " if resume_model is not None else "Новый прогон с нуля. ")
        + f"Оценка каждые {args.eval_every:,} переходов и после финального обновления. "
        f"На каждой оценке {args.eval_episodes} детерминированных и {args.eval_episodes} стохастических эпизодов, "
        f"seeds 10 000–{10000 + args.eval_episodes - 1:,}, "
        "лимит эпизода 5 000 шагов. Генератор боёв фиксируется только при оценке.\n\n"
        f"При {args.n_envs} средах контрольная точка сохраняется на первом векторном шаге, "
        "достигшем порога. Достигнутая волна — бой с её отрядом; "
        "побеждённая волна — уничтожены все её отряды. "
        "Оценка читает сохранённую модель отдельным процессом, обучение продолжается.\n\n"
    )
    evaluation_rows = []
    train_env = {**os.environ, "MPLBACKEND": "Agg"}
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        train_env[name] = str(args.torch_num_threads)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   env=train_env, start_new_session=True)
        status.update(status="training", namespace_pid=process.pid)
        save_json(output / "launch.json", status)
        print(f"STARTED {output}", flush=True)
        next_index = 0
        while True:
            text = log_path.read_text()
            behavior = latest_behavior(behavior_path)
            if behavior:
                status["reported_training_steps"] = behavior["step"]
            if next_index < len(milestones):
                milestone = milestones[next_index]
                final = milestone == args.total_steps
                if final:
                    model_path = output / "models/final_model.zip"
                    vec_path = output / "models/vecnormalize.pkl"
                    ready = "Финальная модель сохранена:" in text
                else:
                    model_path = output / f"checkpoints/campaign_ppo_step_{milestone}.zip"
                    vec_path = output / f"checkpoints/vecnormalize_step_{milestone}.pkl"
                    ready = f"[Checkpoint] Saved at {milestone} steps" in text
                if ready:
                    eval_output = output / "assessments" / f"step_{milestone}"
                    eval_output.mkdir(parents=True, exist_ok=True)
                    status.update(status="evaluating", evaluating_milestone=milestone)
                    save_json(output / "launch.json", status)
                    print(f"EVALUATING milestone={milestone}", flush=True)
                    eval_command = [sys.executable, "-u", str(ROOT / "benchmark_training.py"), "eval",
                                    "--run-dir", str(output), "--model-path", str(model_path),
                                    "--vecnormalize-path", str(vec_path), "--eval-output", str(eval_output),
                                    "--episodes", str(args.eval_episodes), "--seed", "10000"]
                    eval_command += ["--map", args.map_name]
                    if args.objective:
                        eval_command += ["--objective", args.objective]
                    if args.no_boss_roster:
                        eval_command.append("--no-boss-roster")
                    with (eval_output / "eval.log").open("w") as eval_log:
                        evaluation_process = subprocess.run(eval_command, cwd=ROOT, stdout=eval_log,
                                                            stderr=subprocess.STDOUT, check=False)
                    if evaluation_process.returncode:
                        status["evaluation_errors"].append({"milestone": milestone,
                                                             "returncode": evaluation_process.returncode})
                        save_json(output / "launch.json", status)
                        print(f"EVALUATION FAILED milestone={milestone}; retrying in 30 seconds", flush=True)
                        time.sleep(30)
                        continue
                    evaluation = json.loads((eval_output / "evaluation.json").read_text())
                    snapshot = latest_behavior(behavior_path, before=evaluation["model_steps"])
                    story = evaluation_story(evaluation, milestone, snapshot)
                    (eval_output / "report.md").write_text(story + "\n")
                    with (output / "progress.md").open("a") as report:
                        report.write(story + "\n\n")
                    if snapshot:
                        save_json(eval_output / "training_behavior.json", snapshot)
                    evaluation_rows.append({"milestone": milestone, **evaluation})
                    save_json(output / "evaluations.json", evaluation_rows)
                    status["completed_evaluations"].append(milestone)
                    status.pop("evaluating_milestone", None)
                    status["status"] = "training"
                    next_index += 1
                    print(story, flush=True)
            returncode = process.poll()
            if returncode is not None:
                status["training_exit_code"] = returncode
                if returncode != 0:
                    status["status"] = "failed"
                    save_json(output / "launch.json", status)
                    raise SystemExit(returncode)
                if next_index == len(milestones):
                    status.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat(),
                                  reported_training_steps=args.total_steps)
                    save_json(output / "launch.json", status)
                    print(f"COMPLETED {args.total_steps} steps; {len(milestones)} evaluations: {output}", flush=True)
                    break
            match = re.findall(r"\|\s+fps\s+\|\s+(\d+)", text)
            if match:
                status["training_log_fps"] = int(match[-1])
            save_json(output / "launch.json", status)
            time.sleep(5)


if __name__ == "__main__":
    main()
