"""Render benchmark results as CSV, a heatmap, and a Russian report."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    sweep = json.loads((root / "sweep/sweep_results.json").read_text())
    comparison_path = root / "comparison/comparison_results.json"
    comparison = json.loads(comparison_path.read_text()) if comparison_path.exists() else []
    fields = ["n_envs", "threads", "seed", "actual_steps", "fps", "learn_seconds",
              "rollout_seconds", "update_seconds", "process_seconds"]
    with (root / "sweep.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(sweep, key=lambda r: (r["n_envs"], r["threads"])))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    env_counts = [1, 2, 4, 6, 8, 12]
    thread_counts = [1, 2, 4, 8]
    values = np.full((len(env_counts), len(thread_counts)), np.nan)
    for row in sweep:
        values[env_counts.index(row["n_envs"]), thread_counts.index(row["threads"])] = row["fps"]
    fig, ax = plt.subplots(figsize=(7, 6))
    heatmap = ax.imshow(values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(thread_counts)), thread_counts)
    ax.set_yticks(range(len(env_counts)), env_counts)
    ax.set_xlabel("PyTorch threads")
    ax.set_ylabel("SubprocVecEnv environments")
    ax.set_title("Super Last Stand: training FPS (higher is better)\n12,000 transitions, including PPO optimizer updates")
    midpoint = (np.nanmin(values) + np.nanmax(values)) / 2
    for i in range(len(env_counts)):
        for j in range(len(thread_counts)):
            value = values[i, j]
            ax.text(j, i, "—" if np.isnan(value) else f"{value:.0f}",
                    ha="center", va="center", color="white" if value > midpoint else "black")
    fig.colorbar(heatmap, ax=ax, label="Environment transitions / second")
    fig.tight_layout()
    fig.savefig(root / "fps_heatmap.png", dpi=180)
    plt.close(fig)

    groups = {}
    for row in comparison:
        groups.setdefault((row["n_envs"], row["threads"]), []).append(row)
    summary = []
    for (envs, threads), rows in sorted(groups.items()):
        summary.append({"n_envs": envs, "threads": threads, "seeds": [r["seed"] for r in rows],
                        "fps_mean": mean(r["fps"] for r in rows),
                        "fps_std": stdev(r["fps"] for r in rows) if len(rows) > 1 else None,
                        "learn_seconds_mean": mean(r["learn_seconds"] for r in rows),
                        **{f"{mode}_{key}": mean(r["evaluation"][mode][key] for r in rows)
                           for mode in ("deterministic", "stochastic")
                           for key in ("wins", "timeouts", "mean_reward", "mean_waves", "mean_wave_stacks", "mean_enemies")}})
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    lines = ["# Super Last Stand: сравнение конфигураций обучения", "",
             "Результаты относятся к этой машине: 8 логических CPU, PyTorch на CPU.", "",
             "## Методика", "",
             "- Каждый запуск — отдельный процесс; прогоны последовательные, без конкуренции за CPU.",
             "- Режим `SubprocVecEnv`, метод запуска `spawn`; все механики карты, маски действий, VecCheckNan и диагностика сохранены.",
             "- Первичный перебор: 20 конфигураций × 12 000 шагов, seed 42. Это короткий предварительный замер, а не доказательство качества обучения.",
             "- Постоянный объём rollout: `n_envs * n_steps = 12 000`, `batch_size=500`, `n_epochs=5`. Так число обновлений на одинаковое количество данных не меняется.",
             "- FPS = шаги / полное время `model.learn()`, включая последнее обновление PPO. Запуск процессов, сохранение файлов, графики и оценка модели измеряются отдельно.",
             "- Порядок конфигураций перемешан фиксированным seed. Версии зависимостей и SHA-256 исходников записаны в `environment.json`.", "",
             "- Ограничение воспроизводимости обучения: `BattleEnv` создаёт собственный `random.Random()` без seed; обычный `--seed` не фиксирует исходы боёв. Исходная механика обучения сохранена, поэтому повторные прогоны могут различаться. В отдельном оценщике генератор каждого боя явно получает seed из потока тестового эпизода; это обеспечивает повторяемость оценки одной модели.", "",
             "## Первичный перебор", "",
             "| Среды | 1 поток | 2 потока | 4 потока | 8 потоков |",
             "|---:|---:|---:|---:|---:|"]
    for i, envs in enumerate(env_counts):
        cells = ["—" if np.isnan(value) else f"{value:.0f}" for value in values[i]]
        lines.append(f"| {envs} | " + " | ".join(cells) + " |")
    lines += ["", "![FPS](fps_heatmap.png)", "", "Полные цифры: `sweep.csv`, `sweep/sweep_results.json`.", ""]
    if summary:
        plan = json.loads((root / "comparison/comparison_plan.json").read_text())
        lines += ["## Проверка скорости и качества на длинных прогонах", "",
                  f"Каждая конфигурация: {plan['steps']:,} шагов, seeds {plan['seeds']}. После обучения — {plan['episodes']} эпизодов с выбором наиболее вероятного действия и {plan['episodes']} со случайным выбором по вероятностям политики. Тестовые seeds начинаются с 10 000 и одинаковы для всех моделей. Лимит эпизода — 5 000 шагов. Награда при оценке не нормализуется.", "",
                  "Число потоков и число сред не заменяют проверку качества политики: больше FPS само по себе не означает больше побед. Два training seed дают лишь предварительную оценку разброса.", "",
                  "| Среды / потоки | Средний FPS | Время обучения, с | Награда det. | Награда stoch. | Волн det. | Волн stoch. | Побед det. / stoch. на seed |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in summary:
            lines.append(f"| {row['n_envs']} / {row['threads']} | {row['fps_mean']:.0f} | {row['learn_seconds_mean']:.1f} | {row['deterministic_mean_reward']:.2f} | {row['stochastic_mean_reward']:.2f} | {row['deterministic_mean_waves']:.2f} | {row['stochastic_mean_waves']:.2f} | {row['deterministic_wins']:.1f} / {row['stochastic_wins']:.1f} |")
        lines += ["", "### Отдельные запуски", "",
                  "| Среды / потоки | Seed | FPS | Награда det. | Награда stoch. | Волн det. | Волн stoch. |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in sorted(comparison, key=lambda r: (r["n_envs"], r["threads"], r["seed"])):
            det, stoch = row["evaluation"]["deterministic"], row["evaluation"]["stochastic"]
            lines.append(f"| {row['n_envs']} / {row['threads']} | {row['seed']} | {row['fps']:.0f} | {det['mean_reward']:.2f} | {stoch['mean_reward']:.2f} | {det['mean_waves']:.2f} | {stoch['mean_waves']:.2f} |")
        lines += ["", "Сырые результаты и все модели находятся в `comparison/`. В каждой папке сохранены `profile.json`, `train.log`, `behavior.jsonl`, `evaluation.json`, `models/final_model.zip` и `models/vecnormalize.pkl`.", ""]
        fastest = max(summary, key=lambda row: row["fps_mean"])
        recommendation = {
            "map": "super_last_stand", "vec_env": "subproc", "vec_start_method": "spawn",
            "n_envs": fastest["n_envs"], "torch_num_threads": fastest["threads"],
            "ppo_n_steps": 12000 // fastest["n_envs"], "ppo_batch_size": 500,
            "ppo_n_epochs": 5, "tested_total_steps": plan["steps"],
            "benchmark_eval_freq": 0,
            "eval_freq_for_48000_transitions": 48000 // fastest["n_envs"],
            "note": "Fastest tested configuration; quality estimates use only two independent training runs.",
        }
        (root / "recommended_config.json").write_text(json.dumps(recommendation, indent=2) + "\n")
        command = (
            f"python3 train_campaign.py --map super_last_stand --total-steps {plan['steps']} "
            f"--vec-env subproc --vec-start-method spawn --n-envs {fastest['n_envs']} "
            f"--torch-num-threads {fastest['threads']} --ppo-n-steps {12000 // fastest['n_envs']} "
            "--ppo-batch-size 500 --ppo-n-epochs 5 --seed 42 --no-comet "
            f"--eval-freq {48000 // fastest['n_envs']} --checkpoint-freq 48000"
        )
        lines += ["## Рекомендация по скорости", "",
                  f"Из проверенных вариантов самый быстрый на длинных прогонах: **{fastest['n_envs']} сред и {fastest['threads']} потока**, средний FPS **{fastest['fps_mean']:.0f}**. Это рекомендация для данной машины и этой карты; число побед и пройденных волн приведено выше, чтобы отдельно оценить качество.", "",
                  "Команда для следующего запуска с периодической оценкой и сохранением раз в 48 000 переходов:", "",
                  "```bash", command, "```", "",
                  "Для воспроизведения замера скорости используйте `--eval-freq 0`; периодическая оценка добавляет время к обычному запуску. 192 000 шагов выбраны как ровно 16 rollout по 12 000. Для другого бюджета PPO обычно округляет количество шагов вверх до полного rollout.", "",
                  "Машиночитаемые параметры: `recommended_config.json`.", ""]
    lines += ["## Частота оценки в существующем скрипте", "",
              "В текущем `train_campaign.py` аргумент `--eval-freq` передаётся в callback без деления на число сред. Поэтому `--eval-freq 50000` при 8 средах означает оценку раз в 400 000 переходов. В бенчмарке периодическая оценка отключена (`--eval-freq 0`), а модели оцениваются отдельно после обучения. В обычном запуске для оценки раз в 48 000 переходов следует использовать `--eval-freq 6000` при 8 средах или `4000` при 12 средах.", "",
              "## Документация", "",
              "- [SB3: размер rollout у MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)",
              "- [SB3: частота callback при нескольких средах](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)",
              "- [PyTorch: конкуренция процессов и потоков за CPU](https://docs.pytorch.org/docs/2.14/notes/multiprocessing.html)", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
