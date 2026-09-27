"""Append verified completed training runs to the shared plain-text journal."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def summarize(run):
    with zipfile.ZipFile(run / "models/final_model.zip") as archive:
        data = json.loads(archive.read("data"))
    manifest = read_json(run / "manifest.json", {})
    command = read_json(run / "command.json", [])
    log = (run / "train.log").read_text(errors="replace")
    behavior = {}
    if (run / "behavior.jsonl").exists():
        with (run / "behavior.jsonl").open() as stream:
            for line in stream:
                if line.strip():
                    behavior = json.loads(line)
    trainer = command[2] if len(command) > 2 else "не указан"
    shape = data["observation_space"].get("_shape")
    lines = [
        f"Запуск: {run.name}",
        f"Артефакты: {run}",
        f"Код: {manifest.get('cwd', trainer)}",
        f"Наблюдения: {manifest.get('observation_version', 'см. описание')}; размер {shape}",
        f"Шагов в модели: {data['num_timesteps']}; seed: {data.get('seed')}",
    ]
    if behavior:
        results = behavior.get("results_total", {})
        lines.append("Результаты обучения на этом этапе: " + json.dumps(results, ensure_ascii=False))
        reward = behavior.get("reward_mean_window")
        if reward is not None:
            lines.append(f"Reward, последние 100 эпизодов: {reward:.4f}")
    evaluation = run / "eval/evaluations.npz"
    if evaluation.exists():
        with np.load(evaluation, allow_pickle=False) as values:
            if len(values["timesteps"]):
                lines.append(f"Последняя оценка на шаге: {int(values['timesteps'][-1])}")
                for mode, label in (("deterministic", "детерминированная"), ("stochastic", "стохастическая")):
                    key = mode + "_victories"
                    if key in values and len(values[key]) and values[key][-1].size:
                        wins = values[key][-1]
                        rewards = values[mode + "_results"][-1]
                        lines.append(f"Оценка {label}: побед {int(wins.sum())}/{wins.size}, reward {float(rewards.mean()):.4f}")
    urls = list(dict.fromkeys(re.findall(r"https://www\.comet\.com/[\w-]+/[\w-]+/[a-f0-9]{32}", log)))
    lines.append("Comet: " + (", ".join(urls) if urls else "ссылка в журнале запуска отсутствует"))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", type=Path, action="append", default=[])
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.wait:
        while True:
            state = read_json(run / "status.json", {})
            if state.get("status") == "failed":
                raise RuntimeError(f"Training failed: {state.get('error')}")
            if state.get("status") == "complete":
                break
            time.sleep(15)
    marker = f"EXPERIMENT: {run}\n"
    lines = [marker.rstrip(), "Описание: " + args.description]
    for stage in [*args.stage, run]:
        lines.extend(summarize(stage))
    lines.append("Записано UTC: " + datetime.now(timezone.utc).isoformat())
    with args.journal.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        existing = stream.read()
        if marker in existing:
            print("Already recorded:", run.name)
            return
        if not existing:
            stream.write("ВСЕ ЭКСПЕРИМЕНТЫ ОБУЧЕНИЯ\n\nИстория: четыре предыдущих эксперимента из этой переписки; далее новые завершённые запуски.\nПродолжение обучения включается в тот же эксперимент с отдельными результатами этапов.\nReward разных версий среды напрямую не сопоставим.\n")
        stream.write("\n" + "=" * 72 + "\n" + "\n".join(lines) + "\n")
        stream.flush()
    print("Recorded:", run.name)


if __name__ == "__main__":
    main()
